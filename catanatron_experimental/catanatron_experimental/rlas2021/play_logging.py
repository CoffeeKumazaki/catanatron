import os
import pandas as pd
from catanatron.game import Game
from catanatron_experimental.machine_learning.utils import (
  get_matrices_path,
  ensure_dir,
  get_discounted_return,
  get_tournament_return,
  get_victory_points_return,
  DISCOUNT_FACTOR,
)
from catanatron_experimental.rlas2021.features import (
  extract_status,
  extract_actions
)

class PlayLogRecord:

  def __init__(self, turn):
    self.num_turns = turn
    self.player = None
    self.status = None
    self.action = None
    self.reward = {}

class PlayLog:

  def __init__(self):
    self.records = []

def generate_log_callback(play_log_directory):

  data = PlayLog()

  def generate_log(game: Game):

    action = game.state.actions[-1]
    record = PlayLogRecord(game.state.num_turns)
    record.player = action.color
    record.status = extract_status(game, action.color)
    record.action = extract_actions(game, action)
    record.reward["RETURN"]   = get_discounted_return(game, action.color, 1)
    record.reward["DRETURN"]  = get_discounted_return(game, action.color, DISCOUNT_FACTOR)
    record.reward["TRETURN"]  = get_tournament_return(game, action.color, 1)
    record.reward["DTRETURN"] = get_tournament_return(game, action.color, DISCOUNT_FACTOR)
    record.reward["VICTORY"]  = get_victory_points_return(game, action.color)

    data.records.append(record)

    if game.winning_color() is not None:
      flush_log(game, data, play_log_directory)

  return generate_log

def flush_log(game: Game, data: PlayLog, play_log_directory):

  status = []
  actions = []
  rewards = []
  for record in data.records:
    status.append(record.status)
    actions.append(record.action)
    rewards.append(record.reward)

  status_df = pd.DataFrame(status, columns=status[0].keys())
  actions_df = pd.DataFrame(actions, columns=actions[0].keys())
  reward_df = pd.DataFrame(rewards, columns=rewards[0].keys())

  samples_path, board_tensors_path, actions_path, rewards_path = get_matrices_path(
    play_log_directory
  ) 

  ensure_dir(play_log_directory)
  is_first_training = not os.path.isfile(samples_path)
  status_df.to_csv(
      samples_path,
      mode="a",
      header=is_first_training,
      index=False,
      compression="gzip",
  )
  actions_df.to_csv(
      actions_path,
      mode="a",
      header=is_first_training,
      index=False,
      compression="gzip",
  )
  reward_df.to_csv(
      rewards_path,
      mode="a",
      header=is_first_training,
      index=False,
      compression="gzip",
  )
