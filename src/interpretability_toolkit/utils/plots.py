import matplotlib.pyplot as plt
from kornia.morphology import gradient
from skimage.filters import threshold_multiotsu, threshold_otsu

from interpretability_toolkit.utils.tensors import normalize


def plot_images_heatmaps(images, heatmaps, threshold=0.5, border_width=3, cmap="Reds", use_sigmoid=False, titles=None):
    if not isinstance(heatmaps, list):
        heatmaps = [heatmaps]
    b = images.shape[0]
    fig, axs = plt.subplots(b, 1+len(heatmaps), figsize=(10, 5 * b), squeeze=False, frameon=False)
    heatmaps = [h.max(dim=1, keepdim=False).values for h in heatmaps]
    heatmaps = [normalize(h).detach() for h in heatmaps]
    
    if use_sigmoid:
        heatmaps = [h.sigmoid() for h in heatmaps]
    images = normalize(images).detach()
    for j, heat in enumerate(heatmaps):
        for i, img in enumerate(images):
            heat = heat[i]
            if is_inverse_necessary(heat):
                heat = 1 - heat
            if threshold == "otsu":
                t = threshold_otsu(heat.cpu().numpy())
            elif threshold == "multiotsu":
                t = threshold_multiotsu(heat.cpu().numpy(), 4)[-1]
            else:
                t = threshold
            heat_tr = (heat > t).float()
            kernel = heat.new_ones(border_width, border_width)

            heat_border = gradient(heat_tr.unsqueeze(0).unsqueeze(0), kernel).squeeze() > 0

            img = img.permute(1, 2, 0)

        
            axs[i, 0].imshow(img.cpu())
            axs[i, 0].get_xaxis().set_visible(False)
            axs[i, 0].get_yaxis().set_visible(False)
                
            axs[i, 1+j].imshow(img.cpu())

            axs[i, 1+j].imshow(heat.cpu(), alpha=(heat_tr * heat).cpu(), cmap=cmap)
            axs[i, 1+j].imshow(heat.cpu(), alpha=0.15, cmap=cmap)
            axs[i, 1+j].imshow(heat_border.cpu(), alpha=heat_border.float().cpu())

            axs[i, 1+j].get_xaxis().set_visible(False)
            axs[i, 1+j].get_yaxis().set_visible(False)
            
            
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig


def is_inverse_necessary(heatmap):
    if heatmap.mean() > 0.75:
        return True
