from catanatron.game import Game
from catanatron.state_functions import player_key
from catanatron_gym.features import iter_players
from catanatron.models.board import get_edges
from catanatron.models.map import NUM_NODES
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.enums import (
  DEVELOPMENT_CARDS,
  RESOURCES,
  Resource,
  BuildingType,
  ActionType,
  VICTORY_POINT,
)
from catanatron.models.enums import Action

def player_features(game: Game, p_color):
  features = dict()

  for i, color in iter_players(game.state.colors, p_color):
    key = player_key(game.state, color)
    if color == p_color:
      ## 勝ち点.
      features["VPS"] = game.state.player_state[
        key + "_ACTUAL_VICTORY_POINTS"
      ]

      ## 最長交易路.
      features["LONGEST_ROAD_LENGTH"] = game.state.player_state[
        key + "_LONGEST_ROAD_LENGTH"
      ]
      
      ## 使用済騎士数
      features["PLAYED_KNIGHT"] = game.state.player_state[key+"_PLAYED_KNIGHT"]
      break

  return features

def hand_features(game: Game, p_color):
  features = dict()

  for i, color in iter_players(game.state.colors, p_color):
    key = player_key(game.state, color)
    if color == p_color:
      ## 資源
      for resource in RESOURCES:
        features[resource] = game.state.player_state[
          key + f"_{resource}_IN_HAND"
        ]

      ## 発展カード.
      for card in DEVELOPMENT_CARDS:
        features[card] = game.state.player_state[
          key + f"_{card}_IN_HAND"
        ]     
      break
  
  return features

def board_features(game: Game, p_color):

  features = {}
  ## initialize
  for i in range(len(game.state.colors)):
    for node_id in range(NUM_NODES):
      features[f"NODE{node_id}_EMPTY"] = True
      for building in [BuildingType.SETTLEMENT, BuildingType.CITY]:
        features[f"NODE{node_id}_{building.value}_P"] = False
        features[f"NODE{node_id}_{building.value}_O"] = False
    for edge in get_edges():
      features[f"EDGE{edge}_ROAD_E"] = True
      features[f"EDGE{edge}_ROAD_P"] = False
      features[f"EDGE{edge}_ROAD_O"] = False

  for i, color in iter_players(game.state.colors, p_color):

    key = "P" if (color == p_color) else "O"
    
    settlements = tuple(
      game.state.buildings_by_color[color][BuildingType.SETTLEMENT]
    )
    cities = tuple(game.state.buildings_by_color[color][BuildingType.CITY])
    roads = tuple(game.state.buildings_by_color[color][BuildingType.ROAD])

    for node_id in settlements:
      features[f"NODE{node_id}_EMPTY"] = False
      features[f"NODE{node_id}_SETTLEMENT_{key}"] = True
      features[f"NODE{node_id}_CITY_{key}"] = False
    for node_id in cities:
      features[f"NODE{node_id}_EMPTY"] = False
      features[f"NODE{node_id}_SETTLEMENT_{key}"] = False
      features[f"NODE{node_id}_CITY_{key}"] = True
    for edge in roads:
      features[f"EDGE{tuple(sorted(edge))}_ROAD_E"] = False
      features[f"EDGE{tuple(sorted(edge))}_ROAD_{key}"] = True

  return features

feature_extractors = [
  # PLAYER FEATURES =====
  player_features,
  hand_features,
  # BOARD FEATURES =====
  board_features,
]


def extract_status(game: Game, actor_color):
  record = {}
  for extractor in feature_extractors:
    record.update(extractor(game, actor_color))
  return record

def status_vector(game: Game, actor_color):
  record = extract_status(game, actor_color)
  return [float(record[k]) for k in sorted(record.keys())]

prev_playable_actions = 50
def extract_actions(game: Game, action: Action):
  record = {}
  record["Action"] = str(action.action_type)
  if action.value is not None:
    record["Action"] += str(action.value)
  
  return record

def get_feature_size():

  players = [
      SimplePlayer(Color.RED),
      SimplePlayer(Color.BLUE),
      SimplePlayer(Color.WHITE),
      SimplePlayer(Color.ORANGE),
  ]
  game = Game(players)
  record = extract_status(game, players[0].color)
  return len(record)
