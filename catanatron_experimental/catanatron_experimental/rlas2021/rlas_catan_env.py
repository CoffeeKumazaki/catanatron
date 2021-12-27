from catanatron.models.enums import Action, Resource, ActionType
from catanatron.models.board import get_edges


RLAS_H_ACTIONS = {
  "Roll": ActionType.ROLL,
  "BuildRoad": ActionType.BUILD_ROAD,
  "BuildSettlement": ActionType.BUILD_SETTLEMENT,
  "BuildCity": ActionType.BUILD_CITY,
  "BuyDevelopmentCard": ActionType.BUY_DEVELOPMENT_CARD,
  "PlayKnightCard": ActionType.PLAY_KNIGHT_CARD,
  "PlayYearOfPlenty": ActionType.PLAY_YEAR_OF_PLENTY,
  "PlayMonopoly": ActionType.PLAY_MONOPOLY,
  "MaritimeTrade": ActionType.MARITIME_TRADE,
  "EndTurn": ActionType.END_TURN,
}

RLAS_EDGES = [
  tuple(sorted(e) for e in get_edges())
]

RLAS_TRADE = [
  tuple(4 * [i] + [j])
  for i in Resource
  for j in Resource
  if i != j
]

def index_to_action(h_index, l_indices, playable_actions):

  action_key = RLAS_H_ACTIONS.keys()[h_index]
  action_type = RLAS_H_ACTIONS[action_key]

  action_value = None
  if action_type in [ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT]:
    action_value = l_indices[0]
  elif action_type in [ActionType.BUILD_ROAD, ActionType.PLAY_YEAR_OF_PLENTY, ActionType.MARITIME_TRADE]:
    ## road: edge(from) and edge(to)
    ## year of plenty: resource1 and 2
    ## trade: resource(from) and resource(to)
    action_value = tuple(l_indices)
  else:
    pass

  action = None
  for a in playable_actions:
    if a.action_type == action_type and a.action_value == action_value:
      action = a
      break

  assert action is not None
  return action

def action_to_index(action: Action):

  print(action)
  h_index = RLAS_H_ACTIONS.keys().index(action.action_type)

  return h_index

def get_h_actions_size():
  return len(RLAS_H_ACTIONS)