import collections
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
import torch_geometric as torch_g
import numpy as np

width = 2
height = 2
vertex = []
edge_u, edge_v = [], []


def location_to_move(location):
    if len(location) != 2:
        return -1
    h = location[0]
    w = location[1]
    move = h * width + w
    if move not in range(width * height):
        return -1
    return move



for x in range(width):
    for y in range(height):
        u = location_to_move([x,y])
        vertex.append([0])
        v = 0
        if x > 0:
            v = location_to_move([x-1,y])
            edge_u.append(u)
            edge_v.append(v)
            edge_u.append(v)
            edge_v.append(u)
            if y > 0:
                v = location_to_move([x-1,y-1])
                edge_u.append(u)
                edge_v.append(v)
                edge_u.append(v)
                edge_v.append(u)
        if y > 0:
            v = location_to_move([x,y-1])
            edge_u.append(u)
            edge_v.append(v)
            edge_u.append(v)
            edge_v.append(u)
            if x + 1 < width:
                v = location_to_move([x+1,y-1])
                edge_u.append(u)
                edge_v.append(v)
                edge_u.append(v)
                edge_v.append(u)
# vertex = np.array(vertex)


vertex = torch.tensor(vertex+[[0]], dtype = torch.float)
edge = torch.tensor([edge_u, edge_v], dtype = torch.long)
states_graph = torch_g.data.Data(x = vertex, edge_index = edge)
print(vertex)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = torch_g.data.Data(x=x, edge_index=edge_index)
dataset = torch_g.loader.DataLoader([data, states_graph], batch_size = 512, shuffle = True)


t = torch_g.data.Batch.from_data_list([data, data])
print(t)
print(t.num_graphs)