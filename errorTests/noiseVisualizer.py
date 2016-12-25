import nn

data = nn.np.genfromtxt(nn.INPUT_FILE, delimiter=nn.DELIMITER, dtype=int)
for noise in [0, 0.05, 0.07, 0.1, 0.14, 0.2, 0.25]:
    print(nn.get_pattern(
        nn.add_noise([data[0][0:35]], noise)[0],
        nn.INPUT_LAYER_WIDTH,
        nn.INPUT_LAYER_HEIGHT))
    print("-----")
