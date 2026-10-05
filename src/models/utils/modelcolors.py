from matplotlib.colors import to_rgb

model_colors = {
    # baseline:
    'persistence'  : "#888888",
    'seasonalaverage'   : "#595959",    

    'gcnmodel' : 'blue',
    'nodemlpmodel' : 'darkgreen'
}

def color_is_light(color: str, threshold=0.6):
    # perceived luminance (human-vision–weighted)
    r, g, b = to_rgb(color)
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    return luminance > threshold
