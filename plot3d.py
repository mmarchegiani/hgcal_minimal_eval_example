import os, os.path as osp, glob, uuid
from textwrap import dedent
from multiprocessing import Pool

import numpy as np
import torch
import matplotlib, matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})

from cmspepr_hgcal_core.gravnet_model import GravnetModelWithNoiseFilter
from cmspepr_hgcal_core.datasets import taus2021_npz_to_torch_data
from cmspepr_hgcal_core.matching import match

# Clustering parameters; Values used for 2021 results were t_beta=.2, t_d=.5
THRESHOLD_BETA = .2
THRESHOLD_DIST = .5

# For colors, just use the existing XKCD colors in Matplotlib.
# See: https://xkcd.com/color/rgb/
import matplotlib._color_data as mcd
XKCD_COLORS = list(mcd.XKCD_COLORS.values())


def get_clustering(beta: np.array, X: np.array, threshold_beta: float=.1, threshold_dist: float=1.) -> np.array:
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted beta and cluster space coordinates) and the clustering
    parameters threshold_beta and threshold_dist.
    Takes numpy arrays as input.
    """
    n_points = beta.shape[0]
    select_condpoints = beta > threshold_beta
    # Get indices passing the threshold
    indices_condpoints = np.nonzero(select_condpoints)[0]
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[np.argsort(-beta[select_condpoints])]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    for index_condpoint in indices_condpoints:
        # Get the distance of every unassigned node to the current cond point
        d = np.linalg.norm(X[unassigned] - X[index_condpoint], axis=-1)
        # Assign all nodes with d < threshold_dist to this cond_point
        assigned_to_this_condpoint = unassigned[d < threshold_dist]
        clustering[assigned_to_this_condpoint] = index_condpoint
        # Reduce the set of unassigned points
        unassigned = unassigned[~(d < threshold_dist)]
    return clustering


def get_plotly(X, y, color_map, sizes=None):
    """
    Makes a list of Plotly traces based on the passed features and clustering.
    """
    import plotly.graph_objects as go
    data = []
    if sizes is None: sizes = 10. * np.ones_like(y)
    for cluster_index in np.unique(y):
        sel = y == cluster_index
        data.append(go.Scatter3d(
            x=X[sel,7], y=X[sel,5], z=X[sel,6],
            text=[f'{e:.2f}' for e in X[:,0]],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=sizes,
                color=color_map[int(cluster_index)],
                ),
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}<br>e=%{{text}}'
                f'<br>clusterindex={cluster_index}'
                f'<br>'
                )
            ))
    return data


def get_plotly_cluster_space(X, y, color_map, sizes=None):
    """
    Makes a list of Plotly traces based on the passed coordinates and clustering.
    """
    import plotly.graph_objects as go
    data = []
    if sizes is None: sizes = 10. * np.ones_like(y)
    for cluster_index in np.unique(y):
        sel = y == cluster_index
        data.append(go.Scatter3d(
            x=X[sel,0], y=X[sel,1], z=X[sel,2],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=sizes,
                color=color_map[int(cluster_index)],
                ),
            hovertemplate=dedent(f"""\
                x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}
                <br>clusterindex={cluster_index}
                <br>
                """),
            ))
    return data


def side_by_side_html(
    data1, data2,
    width=600, height=None, include_plotlyjs='cdn',
    ):
    """
    Script to grab two plotly data lists, and compile them into a single html string.
    Adds JS code to sync the camera angles between the two plots whenever one of them
    is rotated.
    """
    import plotly.graph_objects as go

    scene = dict(
        xaxis_title='z', yaxis_title='x', zaxis_title='y',
        aspectmode='cube'
        )

    if height is None: height = width

    fig1 = go.Figure(data=data1)
    fig1.update_layout(width=width, height=height, scene=scene)
    fig2 = go.Figure(data=data2)
    fig2.update_layout(width=width, height=height, scene=scene)

    fig1_html = fig1.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
    fig2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

    # Extract the id's of the <div>'s that Plotly generated (hacky)
    divid1 = fig1_html.split('<div id="',1)[1].split('"',1)[0]
    divid2 = fig2_html.split('<div id="',1)[1].split('"',1)[0]

    # Create some new ids
    id1 = str(uuid.uuid4())[:6]
    id2 = str(uuid.uuid4())[:6]

    # Put in the two Plotly html's, and add event handlers to update camera angles
    # when one of the two plots is rotated.
    html = dedent(f"""\
        <div style="width: 47%; display: inline-block">
        {fig1_html}
        </div>
        <div style="width: 47%; display: inline-block">
        {fig2_html}
        </div>
        <script>
        var graphdiv_{id1} = document.getElementById("{divid1}");
        var graphdiv_{id2} = document.getElementById("{divid2}");
        var isUnderRelayout_{id1} = false
        graphdiv_{id1}.on("plotly_relayout", () => {{
            // console.log("relayout", isUnderRelayout_{id1})
            if (!isUnderRelayout_{id1}) {{
                Plotly.relayout(graphdiv_{id2}, {{"scene.camera": graphdiv_{id1}.layout.scene.camera}})
                .then(() => {{ isUnderRelayout_{id1} = false }}  )
                }}
            isUnderRelayout_{id1} = true;
            }})
        var isUnderRelayout_{id2} = false
        graphdiv_{id2}.on("plotly_relayout", () => {{
            // console.log("relayout", isUnderRelayout_{id2})
            if (!isUnderRelayout_{id2}) {{
                Plotly.relayout(graphdiv_{id1}, {{"scene.camera": graphdiv_{id2}.layout.scene.camera}})
                .then(() => {{ isUnderRelayout_{id2} = false }}  )
                }}
            isUnderRelayout_{id2} = true;
            }})
        </script>
        """)
    return html


def make_plots(model, npz_file, threshold_beta=None, threshold_dist=None, device='cpu'):
    if threshold_beta is None:
        threshold_beta = THRESHOLD_BETA
    if threshold_dist is None:
        threshold_dist = THRESHOLD_DIST
    
    data = taus2021_npz_to_torch_data(npz_file)
    data.batch = torch.ones(data.x.size(0), dtype=torch.long)
    
    # Move data to device
    data = data.to(device)
    print(data)

    x = data.x.cpu().numpy()
    energy = x[:,0]
    y_true = data.y.cpu().numpy()
    
    with torch.no_grad():
        score_noise_filter, pass_noise_filter, out_gravnet = model(data)

    n_pass = pass_noise_filter.sum()
    n_total = len(pass_noise_filter)
    n_filter = n_total - n_pass
    print(f'Noise filter filtering away {100.*n_filter/n_total:.3f}% of events')

    # Quick histogram plot of the noise filter score
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()
    bins = np.linspace(0., 1., 100)
    hist, _, _ = ax.hist(torch.exp(score_noise_filter[:, 1]).cpu(), bins=bins, label='Noise filter score')
    ax.plot(2*[model.signal_threshold], [0., max(hist)], label='Threshold')
    ax.legend()
    plot_filename = f'tmp_beta{threshold_beta:.2f}_dist{threshold_dist:.2f}.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    # os.system('imgcat tmp.png') # Display image in terminal; Only if you use iTerm2 and have imgcat on your path

    # First column of the output is the object condensation beta; don't forget the sigmoid
    beta = torch.sigmoid(out_gravnet[:,0]).cpu().numpy()
    # All other columns are the cluster space coordinates
    cluster_space_coords = out_gravnet[:,1:].cpu().numpy()

    # Determine which nodes belong to which cond point according to the model.
    y_pred_pnf = get_clustering(beta, cluster_space_coords, threshold_beta, threshold_dist)

    # This y_pred_pnf is only valid for hits that *P*assed the *N*oise *F*ilter.
    # At this point, len(y_pred) == len(out_gravnet) < len(y_true)
    # Make a new y_pred now, so that len(y_true) == len(y_pred)
    y_pred = np.zeros_like(y_true)
    y_pred[pass_noise_filter.cpu()] = y_pred_pnf

    # Match predicted to truth
    matches = match(y_true, y_pred, energy, threshold=0.2)

    # Make a color map per cluster
    colors = XKCD_COLORS[:]
    # Shuffle to avoid similar colors close together
    np.random.default_rng(1001).shuffle(colors)
    # Assign light grey to 0 and -1
    color_map_true = {0: '#bfbfbf', -1: '#bfbfbf'}
    color_map_pred = {0: '#bfbfbf', -1: '#bfbfbf'}
    for i_truth, i_pred, _ in zip(*matches):
        if i_truth in color_map_true and i_pred in color_map_pred:
            continue
        elif i_truth in color_map_true:
            color_map_pred[i_pred] = color_map_true[i_truth]
        elif i_pred in color_map_pred:
            color_map_true[i_truth] = color_map_pred[i_pred]
        else:
            color = colors.pop()
            color_map_true[i_truth] = color
            color_map_pred[i_pred] = color
    # Also assign colors for unmatched indices
    for i in np.unique(y_true):
        if not i in color_map_true: color_map_true[i] = colors.pop()
    for i in np.unique(y_pred):
        if not i in color_map_pred: color_map_pred[i] = colors.pop()

    # Compute dot sizes in the plot based on energy
    sizes = (energy - energy.mean()) / energy.std()
    sizes = 1 / (1. + np.exp(sizes)) # sigmoid
    sizes += 1. + 2.*sizes # Scale to sizes

    # Also make a plot of cluster space coordinates
    # First need to bring the 5D clustering space down to 3D, use PCA for that
    if cluster_space_coords.shape[1] > 3:
        from sklearn.decomposition import PCA
        cluster_space_coords = PCA(3).fit_transform(cluster_space_coords)

    # Compile a .html file with the plots in it
    html_filename = f'myplots_beta{threshold_beta:.2f}_dist{threshold_dist:.2f}.html'
    with open(html_filename, 'w') as f:
        f.write(dedent(f"""\
            <p>Endcap: {data.endcap}</p>
            <p>Threshold Beta: {threshold_beta}, Threshold Distance: {threshold_dist}</p>
            <div style="display:flex">
              <div style="flex:50%">
                <h2>Predicted clustering</h2>
                </div>
              <div style="flex:50%">
                <h2>Truth clustering</h2>
                </div>
            </div>
            """))
        f.write(side_by_side_html(
            get_plotly(x, y_pred, color_map_pred, sizes),
            get_plotly(x, y_true, color_map_true, sizes)
            ))
        f.write(dedent(f"""
            <div style="display:flex">
              <div style="flex:50%">
                <h2>Clustering space: colored by prediction</h2>
                </div>
              <div style="flex:50%">
                <h2>Clustering space: colored by truth</h2>
                </div>
            </div>
            """))
        f.write(side_by_side_html(
            get_plotly_cluster_space(cluster_space_coords, y_pred[pass_noise_filter.cpu()], color_map_pred, sizes),
            get_plotly_cluster_space(cluster_space_coords, y_true[pass_noise_filter.cpu()], color_map_true, sizes)
            ))

        # Hacky: Include the noise filter histogram directly into the html file as a
        # base64 string.
        import base64
        with open(plot_filename, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        f.write(f'\n<img alt="Noise filter score" src="data:image/png;base64,{encoded_string}" />')
    
    print(f'Generated plots for beta={threshold_beta:.2f}, dist={threshold_dist:.2f}: {html_filename}')


def make_plots_wrapper(args):
    """
    Wrapper function for multiprocessing.
    Args should be a tuple: (model_state_dict, npz_file, threshold_beta, threshold_dist, device_id)
    """
    model_state_dict, npz_file, threshold_beta, threshold_dist, device_id = args
    
    # Determine device for this worker
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() and device_id is not None else 'cpu')
    
    # Recreate model for this worker (each process needs its own model instance)
    model = GravnetModelWithNoiseFilter(
        input_dim=9,
        output_dim=6,
        k=50,
        signal_threshold=.05
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return make_plots(model, npz_file, threshold_beta, threshold_dist, device)


def main():
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load weights into model
    ckpt = 'ckpt_train_taus_integrated_noise_Oct20_212115_best_397.pth.tar'
    model = GravnetModelWithNoiseFilter(
        input_dim = 9,
        output_dim = 6,
        k=50,
        signal_threshold=.05
        )
    model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
    model.to(device)
    model.eval()

    # One file example now
    npz_files = glob.glob('events/*.npz')
    if not npz_files:
        print("No .npz files found in 'events/' directory")
        return
    
    npz_file = npz_files[0]
    
    # Define parameter ranges for grid search
    threshold_betas = [0.1, 0.2, 0.3]
    threshold_dists = [0.3, 0.5, 0.7]
    
    # Create parameter combinations
    param_combinations = [(beta, dist) for beta in threshold_betas for dist in threshold_dists]
    
    # Determine number of available GPUs and processes
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_processes = max(1, num_gpus) if num_gpus > 0 else 8  # Use CPU cores if no GPU
    
    print(f'Running {len(param_combinations)} parameter combinations on {num_processes} processes')
    print(f'Parameter combinations: {param_combinations}')
    
    # Prepare arguments for multiprocessing
    model_state_dict = model.state_dict()
    args_list = []
    
    for i, (threshold_beta, threshold_dist) in enumerate(param_combinations):
        # Distribute work across available devices
        device_id = i % num_gpus if num_gpus > 0 else None
        args_list.append((model_state_dict, npz_file, threshold_beta, threshold_dist, device_id))
    
    # Use multiprocessing to run in parallel
    with Pool(processes=num_processes) as pool:
        pool.map(make_plots_wrapper, args_list)
    
    print("All parameter combinations completed!")


if __name__ == '__main__':
    main()