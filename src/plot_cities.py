import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_cities(csv_file):
    """
    Load cities from a CSV file
    """
    cities_df = pd.read_csv(csv_file)
    print(f"Loaded {len(cities_df)} cities from {csv_file}")
    return cities_df

def plot_cities(cities_df=None, csv_file=None, show_indices=True, save_plot=False, filename='cities_plot.png'):
    """
    Plot the cities
    """
    # Load cities from CSV if not provided as DataFrame
    if cities_df is None and csv_file is not None:
        cities_df = load_cities(csv_file)
    elif cities_df is None:
        raise ValueError("Either cities_df or csv_file must be provided")
    
    # Get distance unit from the DataFrame attributes
    distance_unit = cities_df.attrs.get('distance_unit', 'units')
    
    plt.figure(figsize=(10, 8))
    plt.scatter(cities_df['x'], cities_df['y'], c='blue', s=100)
    
    if show_indices:
        for i, row in cities_df.iterrows():
            plt.annotate(str(row['city_id']), 
                         (row['x'] + 0.5, row['y'] + 0.5),
                         fontsize=12)
    
    plt.title('Cities for TSP')
    plt.xlabel(f'X Coordinate ({distance_unit})')
    plt.ylabel(f'Y Coordinate ({distance_unit})')
    plt.grid(True)
    
    if save_plot:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    
    plt.show()


def plot_paths(cities_df, paths, show_indices=True, save_plot=False, filename='paths_plot.png', 
               base_coords=None, return_to_base=False):
    """
    Plot multiple paths between cities
    
    Args:
        cities_df (DataFrame): DataFrame containing city information with 'city_id', 'x', and 'y' columns
        paths (list): List of lists, where each inner list contains a sequence of city_ids representing a path
        show_indices (bool): Whether to show city indices on the plot
        save_plot (bool): Whether to save the plot to a file
        filename (str): Filename to save the plot if save_plot is True
        base_coords (tuple): Coordinates of the base location (x, y) if return_to_base is True
        return_to_base (bool): Whether paths should return to the base location
    """
    # Get distance unit from the DataFrame attributes
    distance_unit = cities_df.attrs.get('distance_unit', 'units')
    
    plt.figure(figsize=(12, 10))
    
    # Plot all cities
    plt.scatter(cities_df['x'], cities_df['y'], c='gray', s=50, alpha=0.5)
    
    # Plot base location if return_to_base is True
    if return_to_base and base_coords is not None:
        plt.scatter(base_coords[0], base_coords[1], c='black', s=100, marker='*', label='Base')
    
    # Colors for different paths
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
    
    # Plot each path
    for i, path in enumerate(paths):
        path_color = colors[i % len(colors)]
        
        # Get coordinates for the path
        x_coords = cities_df.loc[path, 'x'].values
        y_coords = cities_df.loc[path, 'y'].values
        
        # If returning to base, add base coordinates to start and end of path
        if return_to_base and base_coords is not None and len(path) > 0:
            x_coords = np.concatenate(([base_coords[0]], x_coords, [base_coords[0]]))
            y_coords = np.concatenate(([base_coords[1]], y_coords, [base_coords[1]]))
        
        # Calculate path distance
        path_distance = 0
        for j in range(len(x_coords) - 1):
            dx = x_coords[j+1] - x_coords[j]
            dy = y_coords[j+1] - y_coords[j]
            path_distance += np.sqrt(dx**2 + dy**2)
        
        # Plot the path
        plt.plot(x_coords, y_coords, 'o-', color=path_color, linewidth=2, 
                 label=f'Path {i+1}: {len(path)} cities, {path_distance:.2f} {distance_unit}')
        
        # Add city labels if requested
        if show_indices:
            for j, city_id in enumerate(path):
                plt.annotate(str(city_id), (cities_df.loc[city_id, 'x'], cities_df.loc[city_id, 'y']), fontsize=10)
    
    plt.title('Travel Paths Between Cities')
    plt.xlabel(f'X Coordinate ({distance_unit})')
    plt.ylabel(f'Y Coordinate ({distance_unit})')
    plt.legend()
    plt.grid(True)
    
    if save_plot:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    
    plt.show()
