import os
import time
import random
import sys, traceback
from pathlib import Path
import click
from collections import Counter, deque

import numpy as np
np.set_printoptions(linewidth=np.inf)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tqdm import tqdm
import selenium
from selenium import webdriver

from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron_experimental.machine_learning.players.minimax import (
    AlphaBetaPlayer,
    ValueFunctionPlayer,
)
from catanatron.state_functions import (
    player_key
)
from catanatron_experimental.rlas2021.features import (
    extract_status, 
    status_vector,
    get_feature_size,
    extract_actions,
)
from catanatron_server.utils import ensure_link
from catanatron_experimental.rlas2021.rlas_catan_env import (
    index_to_action,
    action_to_index,
)
from catanatron_gym.envs.catanatron_env import (
    ACTIONS_ARRAY,
    ACTION_SPACE_SIZE,
    from_action_space,
    to_action_space,
)
from catanatron.models.map import BaseMap

from catanatron_experimental.machine_learning.utils import (
  generate_arrays_from_file,
  estimate_num_samples,
)


DISCOUNT = 0.9

# Every 5 episodes we have ~MINIBATCH_SIZE=1024 samples.
# With batch-size=16k we are likely to hit 1 sample per action(?)
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 2_000  # Min number of steps in a memory to start training
MINIBATCH_SIZE = 1024  # How many steps (samples) to use for training
TRAIN_EVERY_N_EPISODES = 1
TRAIN_EVERY_N_STEPS = 100  # catan steps / decisions by agent
UPDATE_MODEL_EVERY_N_TRAININGS = 5  # Terminal states (end of episodes)

# Environment exploration settings
# tutorial settings (seems like 26 hours...)
# EPISODES = 20_000
# EPSILON_DECAY = 0.99975
# 8 hours process
# EPISODES = 6000
EPSILON_DECAY = 0.9993
# 2 hours process
# EPISODES = 1500
# EPSILON_DECAY = 0.998
# 30 mins process
EPISODES = 150
# EPSILON_DECAY = 0.98
# EPISODES = 10_000
epsilon = 1  # not a constant, going to be decayed
MIN_EPSILON = 0.001

# Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = False

# TODO: Simple Action Space:
# Hold
# Build Settlement on most production spot vs diff. need number to translate enemy potential to true prod.
# Build City on most production spot.
# Build City on spot that balances production the most.
# Build Road towards more production. (again need to translate potential to true.)
# Buy dev card
# Play Knight to most powerful spot.
# Play Year of Plenty towards most valueable play (city, settlement, dev). Bonus points if use rare resources.
# Play Road Building towards most increase in production.
# Play Monopoly most impactful resource.
# Trade towards most valuable play.

# TODO: Simple State Space:
# Cards in Hand
# Buildable Nodes
# Production
# Num Knights
# Num Roads

FEATURES_SIZE = get_feature_size()
ACTIONS_SIZE = ACTION_SPACE_SIZE# get_actions_size()

class RLASDQNPlayer(Player):
    def __init__(self, color, model_path):
        super(RLASDQNPlayer, self).__init__(color)
        self.model = tf.keras.models.load_model(model_path)   

    def decide(self, game, playable_actions):
        # 選択肢がひとつのときはそれをする
        if len(playable_actions) == 1:
            return playable_actions[0]

        # get status vector
        sample = status_vector(game, self.color)
        sample = tf.reshape(tf.convert_to_tensor(sample), (-1, FEATURES_SIZE))
        # get estimated q value
        qs = self.model.call(sample)[0]

        best_action_int = epsilon_greedy_policy(playable_actions, qs, 0.05)
        best_action = from_action_space(best_action_int, playable_actions)
        # print(best_action_int, best_action)
        return best_action

BOT_CLASSES = [
    RandomPlayer,
    WeightedRandomPlayer,
    VictoryPointPlayer,
    ValueFunctionPlayer,
    AlphaBetaPlayer,
]

class CatanEnvironment:
    def __init__(self):
        self.game = None
        self.dqn_players = []

    def playable_actions(self):
        return self.game.state.playable_actions

    def reset(self):
        camap = BaseMap("beginner")
        ## choose number of players and AI for each player randomly.
        # DQN Player + 1-3 players
        player_num = random.randrange(1, 4) + 1
        dqn_player_num = random.randrange(0, player_num) + 1
        pcolors = [c for c in Color]

        ## print(f"{dqn_player_num} DQNs / {player_num} players")

        # dqn player
        players = []
        for i in range(dqn_player_num):
            p = Player(pcolors.pop())
            players.append(p)
            self.dqn_players.append(p.color)

        for i in range(player_num-dqn_player_num):
            bot_id = random.randrange(0,len(BOT_CLASSES))
            players.append(BOT_CLASSES[bot_id](pcolors.pop()))

        ## for p in players:
        ##    print(p)

        game = Game(players=players, catan_map=camap)
        self.game = game

        return self._get_state()

    def step(self, action_int):
        action = from_action_space(action_int, self.playable_actions())
        key = player_key(self.game.state, action.color)
        ppoint = self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]

        self.game.execute(action)

        winning_color = self.game.winning_color()

        new_state = self._get_state()

        reward = self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] - ppoint
        # reward = int(winning_color == action.color) * 10 * 1000 + points
        #if winning_color is None:
        #    reward = 0
        #else:
        #    reward = 1

        done = winning_color is not None or self.game.state.num_turns > 500
        return new_state, reward, done

    def render(self):
        driver = webdriver.Chrome()
        link = ensure_link(self.game)
        driver.get(link)
        time.sleep(1)
        try:
            driver.close()
        except selenium.common.exceptions.WebDriverException as e:
            print("Exception closing browser. Did you close manually?")

    def _get_state(self):
        current_player_color = self.game.state.current_player().color
        sample = status_vector(self.game, current_player_color)

        return (sample, None)  # NOTE: each observation/state is a tuple.

    def _advance_until_dqn_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_player().color not in self.dqn_players
        ):
            self.game.play_tick()  # will play bot


# Agent class
class RLASAgent:
    def __init__(self, model_path):
        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
            self.target_model = tf.keras.models.load_model(model_path)
        else:
            # Main model
            self.model = self.create_model()

            # Target network
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        
        inputs = tf.keras.Input(shape=(FEATURES_SIZE,))
        outputs = inputs
        # outputs = normalizer_layer(outputs)
        outputs = BatchNormalization()(outputs)
        # outputs = Dense(352, activation="relu")(outputs)
        # outputs = Dense(256, activation="relu")(outputs)
        outputs = Dense(64, activation="relu")(outputs)
        outputs = Dense(32, activation="relu")(outputs)
        outputs = Dense(units=ACTIONS_SIZE, activation="linear")(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=1e-5),
            metrics=["accuracy"],
        )

        return model

    # リプレイバッファへの保存
    # (state, action, reward, new state, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # 学習
    def train(self, terminal_state):
        # データ数が MIN_REPLAY_MEMORY_SIZE より少なかったら学習しない
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            print("Not enough training data", len(self.replay_memory), MINIBATCH_SIZE)
            return

        # リプレイバッファからランダムにサンプルを取得
        # TODO: 学習価値の低いアクションも選ばれてしまう(ダイスを振る, ターンを終了するなど)
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # 現在状態のリスト
        current_states = tf.convert_to_tensor([t[0][0] for t in minibatch])
        # 現在状態のQ値のリスト
        current_qs_list = self.model.call(current_states).numpy()

        # 次状態のリスト
        new_current_states = tf.convert_to_tensor([t[3][0] for t in minibatch])
        # 次状態のQ値のリスト
        # 注:ターゲットネットワークから取る
        future_qs_list = self.target_model.call(new_current_states).numpy()

        # 学習データ
        # X: 現在の状態
        # Y: Q値（今回とった行動のQ値に得られたrewardを加える)
        X = []
        y = []
        action_ints = list(map(lambda b: b[1], minibatch))
        action_ints_counter = Counter(action_ints)
        sample_weight = []
        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):
            # 今回の行動に対する報酬を追加
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # 今回の行動に対する報酬で更新
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state[0])
            y.append(current_qs)
            # 行動数の偏りを重みでならす
            w = 1 / (action_ints_counter[action])
            sample_weight.append(w)

        self.model.fit(
            tf.convert_to_tensor(X),
            tf.convert_to_tensor(y),
            sample_weight=np.array(sample_weight),
            batch_size=MINIBATCH_SIZE,
            epochs=1,
            verbose=0,
            shuffle=False,  # no need since minibatch already was a random sampling
        )

        # ゲームが終了したらカウンターをinc
        if terminal_state:
            self.target_update_counter += 1

        # UPDATE_MODEL_EVERY_N_TRAININGS 回数だけゲームが終わったらモデルを更新.
        if self.target_update_counter > UPDATE_MODEL_EVERY_N_TRAININGS:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            print("Updated model!")

    # 現在状態の推定Q値取得
    def get_qs(self, state):
        (sample, board_tensor) = state
        sample = tf.reshape(tf.convert_to_tensor(sample), (-1, FEATURES_SIZE))
        return self.model.call(sample)[0]


def epsilon_greedy_policy(playable_actions, qs, epsilon):
    if np.random.random() > epsilon:
        # create mask list, 1 if playable else 0
        action_ints = list(map(to_action_space, playable_actions))
        mask = np.zeros(ACTIONS_SIZE, dtype=np.int)
        mask[action_ints] = 1

        clipped_probas = np.multiply(mask, qs)
        clipped_probas[clipped_probas == 0] = -np.inf

        best_action_int = np.argmax(clipped_probas)
        # with open("/app/qs.log", 'a') as f:
            # qs.numpy().tofile(f, sep=',')
            # np.savetxt(f, [qs.numpy()], delimiter=',')
        # print(clipped_probas)
    else:
        # Get random action
        index = random.randrange(0, len(playable_actions))
        best_action = playable_actions[index]
        best_action_int = to_action_space(best_action)

    return best_action_int


def self_learning(agent, metrix_writer, num_episode, output_model_path):
 
  global epsilon
  env = CatanEnvironment()

  # For stats
  ep_rewards = []

  for episode in tqdm(range(1, num_episode + 1), ascii=True, unit="episodes"):

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        best_action_int = epsilon_greedy_policy(
            env.playable_actions(), agent.get_qs(current_state), epsilon
        )
        new_state, reward, done = env.step(best_action_int)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # 行動をバッファに登録
        # 選択肢がないときは学習しなくていい
        if len(env.playable_actions()) > 1 and best_action_int != ACTION_SPACE_SIZE-1 and best_action_int != 0:
          agent.update_replay_memory(
              (current_state, best_action_int, reward, new_state, done)
          )

        if step % TRAIN_EVERY_N_STEPS == 0:
            agent.train(done)

        current_state = new_state
        step += 1
    if step % TRAIN_EVERY_N_EPISODES == 0:
        agent.train(done)

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if episode % AGGREGATE_STATS_EVERY == 0:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
            ep_rewards[-AGGREGATE_STATS_EVERY:]
        )
        with metrix_writer.as_default():
            tf.summary.scalar("avg-reward", average_reward, step=episode)
            tf.summary.scalar("epsilon", epsilon, step=episode)
            metrix_writer.flush()

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    if episode % 100 == 0:
        print("Saving model to", output_model_path, "at ", episode)
        agent.model.save(output_model_path + "-" + '{:04}'.format(episode))

  return agent

def teacher_learning(agent, writer, play_data_path, validation_step):

  num_samples = estimate_num_samples(play_data_path)
  print("Number of samples =", num_samples)

  agent.model.fit(
    generate_arrays_from_file(play_data_path, MINIBATCH_SIZE, "VICTORY_POINTS_RETURN", "P"),
    epochs=1,
    steps_per_epoch = num_samples / MINIBATCH_SIZE,
  )

  return agent


@click.command()
@click.argument("experiment_name")
@click.option(
    "-m",
    "--base-model",
    default=None,
    help="Base model path",
)
@click.option(
  "-d",
  "--play-data-path",
  default=None,
  help="Teacher data path. if empty self-learning",
)
@click.option(
  "-e",
  "--episode",
  default=150,
  help="Number of episode for self-learning",
)
@click.option(
  "--validation-step",
  default=100,
  help=""
)
def main(experiment_name, base_model, play_data_path, episode, validation_step):

    # For more repetitive results
    # random.seed(4)
    # np.random.seed(4)
    # tf.random.set_seed(4)

    # Ensure models folder
    model_name = f"{experiment_name}-{int(time.time())}"
    models_folder = "data/models/"
    if not os.path.isdir(models_folder):
        os.makedirs(models_folder)

    agent = RLASAgent(base_model)
    agent.model.summary()
    metrics_path = f"data/logs/catan-dql/{model_name}"
    output_model_path = models_folder + model_name
    writer = tf.summary.create_file_writer(metrics_path)
    print("Will be writing metrics to", metrics_path)
    print("Will be saving model to", output_model_path)

    if play_data_path is not None:
      agent = teacher_learning(agent, writer, play_data_path, validation_step)
    else:
      agent = self_learning(agent, writer, episode, output_model_path)

    print("Saving model to", output_model_path)
    agent.model.save(output_model_path)


if __name__ == "__main__":
    main()
