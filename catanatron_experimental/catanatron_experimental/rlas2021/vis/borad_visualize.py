import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
import matplotlib.animation as animation

def visualize_tiles(radius):

  tile_color = {
    "Water": "skyblue",
    "Port": "skyblue",
    "SHEEP": "greenyellow",
    "ORE": "dimgray",
    "WOOD": "forestgreen",
    "WHEAT": "gold",
    "BRICK": "saddlebrown",
    "DESERT": "cornsilk",
  }

  fig = plt.figure(figsize=(10, 10)) 
  ax = fig.add_subplot(111)
  # ax.patch.set_facecolor('skyblue')
  ax.set_xlim(-7, 7)
  ax.set_ylim(-7, 7)
  ax.axis("off")
  ax.patch.set_alpha(0.0)
  ## show tile
  for key in tiles.keys():
    p, po = get_tile_polygon(key, radius)
    # plt.plot(p[0], p[1], 'ro') 
    tile_type = tiles[key].split(".")[0]
    poly = plt.Polygon(po,fc=tile_color[tile_type])

    # show port
    if (tile_type == "Port"):
      dir = tiles[key].split(".")[1]
      try:
        col = tile_color[tiles[key].split(".")[2]]
      except:
        col = "black"

      if (dir == "WEST"):
        port_pos = p + (radius*math.sqrt(3), 0)
        pier_pos =  [po[4], po[5]]
      elif (dir == "EAST"):
        port_pos = p + (-radius*math.sqrt(3), 0)
        pier_pos = [po[1], po[2]]
      elif (dir == "NORTHWEST"):
        port_pos = p + (radius, radius*math.sqrt(3)/2.0)
        pier_pos = [po[0], po[5]]
      elif (dir == "NORTHEAST"):
        port_pos = p + (radius, -radius*math.sqrt(3)/2.0)
        pier_pos = [po[1], po[0]]
      elif (dir == "SOUTHWEST"):
        port_pos = p + (-radius, radius*math.sqrt(3)/2.0)
        pier_pos = [po[3], po[4]]
      elif (dir == "SOUTHEAST"):
        port_pos = p + (-radius, -radius*math.sqrt(3)/2.0)
        pier_pos = [po[3], po[2]]
      else:
        port_pos = None

      if (port_pos is not None):
        plt.plot([port_pos[0], pier_pos[0][0]], [port_pos[1], pier_pos[0][1]], color="gray", ls="dotted")
        plt.plot([port_pos[0], pier_pos[1][0]], [port_pos[1], pier_pos[1][1]], color="gray", ls="dotted")
        plt.plot(port_pos[0], port_pos[1], 'o', color=col, markersize=10)

    ax.add_patch(poly)

  return fig, ax

def visualize_board(actions, game_index = 0, turn = np.inf):

  gameids = actions["Game"].unique()
  actions = actions[actions["Game"]==gameids[game_index]]

  radius = 1.0
  fig, ax = visualize_tiles(radius)

  ims = []
  cturn = 0
  ## show buildings
  ncx, ncy = get_node_coordinates(radius)
  for _, action in actions.iterrows():

    player_color = action["PLAYER"].split(".")[1].lower()
    if player_color == "orange":
      player_color = "darkorange"

    action_type = action["Action"]
    if "BUILD_SETTLEMENT" in action_type:
      node_id = int(action_type.replace("ActionType.BUILD_SETTLEMENT", ""))
      markersize = 10
      x = ncx[node_id]
      y = ncy[node_id]
      s = 'h'
      z = 3
    elif "BUILD_CITY" in action_type:
      node_id = int(action_type.replace("ActionType.BUILD_CITY", ""))
      markersize = 15
      x = ncx[node_id]
      y = ncy[node_id]
      s = 'h'
      z = 3
    elif "BUILD_ROAD" in action_type:
      ids = action_type.replace("ActionType.BUILD_ROAD", "").split(",")
      edge_id_1 = int(ids[0].replace("(", ""))
      edge_id_2 = int(ids[1].replace(")", ""))
      x = [ncx[edge_id_1], ncx[edge_id_2]]
      y = [ncy[edge_id_1], ncy[edge_id_2]]
      s = "-"
      z = 2
    else:
      continue

    plt.plot(x, y, s, color=player_color, markersize=markersize, zorder=z, linewidth=3)
    cturn += 1
    if cturn > turn:
      break

  plt.show()

def get_tile_polygon(coordinate, radius):
  
  '''
   ／ 0 ＼
  5       1
  |        |
  4       2
   ＼ 3 ／
  '''

  ecode = [ 
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0)
  ]
  
  x = math.sqrt(3) * radius * ( coordinate[2]/2 + coordinate[0] )
  y = 3/2 * radius * -coordinate[2]

  ecode[0] = (x, y + radius)
  ecode[1] = (x + radius * math.sqrt(3)/2, y + radius/2)
  ecode[2] = (x + radius * math.sqrt(3)/2, y - radius/2)
  ecode[3] = (x, y - radius)
  ecode[4] = (x - radius * math.sqrt(3)/2, y - radius/2)
  ecode[5] = (x - radius * math.sqrt(3)/2, y + radius/2)

  for i in range(6):
    ecode[i] = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, ecode[i]))

  return (x, y) ,tuple(ecode)

def get_node_coordinates(radius):
  ncs = []

  for key in tiles.keys():
    _, po = get_tile_polygon(key, radius)

    for npos in po:
      if (npos not in ncs):
        ncs.append(npos)
  
  ncx = []
  ncy = []
  for nc in ncs:
    ncx.append(nc[0])
    ncy.append(nc[1])

  return ncx, ncy
