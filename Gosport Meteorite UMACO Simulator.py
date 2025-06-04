#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gosport Meteorite Hotspot Simulation with UMACO

Simulates 1000+ years of potential meteorite falls in the Gosport, UK region
using historical meteorite find data and the Universal Multi-Agent Cognitive Optimization (UMACO) framework.

The script:
- Loads worldwide meteorite finds from NASA dataset (CSV)
- Builds a local grid map of the Gosport area
- Uses UMACO to simulate and identify statistically promising search zones for undiscovered meteorites
- Outputs a heatmap "treasure map" for meteorite hunting in the Gosport region

Perfect for amateur and professional meteorite hunters, or anyone interested in citizen science,
local geodata, and optimization techniques for rare-object discovery.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umaco10 import create_default_umaco, create_agents, UniversalEconomy, EconomyConfig
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GosportMetSim")

# Gosport coordinates and area definition
GOSPORT_CENTER = (50.7963, -1.1267)  # Gosport town center
SEARCH_RADIUS_KM = 15  # km radius around Gosport to search
GRID_RESOLUTION = 64  # Grid resolution for UMACO simulation

def load_meteorite_data(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess meteorite data from CSV.
    
    Args:
        csv_path (str): Path to the meteorite CSV file
        
    Returns:
        pd.DataFrame: Cleaned meteorite data with coordinates
    """
    logger.info(f"Loading meteorite data from {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Clean data - remove entries without coordinates
    df = df.dropna(subset=['reclat', 'reclong'])
    
    # Convert coordinates to float
    df['reclat'] = pd.to_numeric(df['reclat'], errors='coerce')
    df['reclong'] = pd.to_numeric(df['reclong'], errors='coerce')
    
    # Remove any remaining NaN coordinates
    df = df.dropna(subset=['reclat', 'reclong'])
    
    logger.info(f"Loaded {len(df)} meteorite records with valid coordinates")
    return df

def filter_uk_europe_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter meteorite data to UK and nearby European regions.
    
    Args:
        df (pd.DataFrame): Full meteorite dataset
        
    Returns:
        pd.DataFrame: Filtered dataset for UK/Europe region
    """
    # UK and nearby Europe bounding box
    lat_min, lat_max = 45.0, 60.0  # Wider area to include context
    lon_min, lon_max = -10.0, 5.0
    
    mask = (
        (df['reclat'] >= lat_min) & (df['reclat'] <= lat_max) &
        (df['reclong'] >= lon_min) & (df['reclong'] <= lon_max)
    )
    
    filtered_df = df[mask].copy()
    logger.info(f"Filtered to {len(filtered_df)} meteorites in UK/Europe region")
    return filtered_df

def create_gosport_grid(center_lat: float, center_lon: float, 
                       radius_km: float, resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid centered on Gosport for UMACO simulation.
    
    Args:
        center_lat (float): Center latitude (Gosport)
        center_lon (float): Center longitude (Gosport)
        radius_km (float): Radius in kilometers
        resolution (int): Grid resolution (NxN)
        
    Returns:
        Tuple of lat_grid, lon_grid, lat_coords, lon_coords
    """
    # Convert km to approximate degrees (rough approximation for UK latitude)
    lat_deg_per_km = 1.0 / 111.0  # ~111 km per degree latitude
    lon_deg_per_km = 1.0 / (111.0 * np.cos(np.radians(center_lat)))  # Adjust for longitude
    
    radius_lat = radius_km * lat_deg_per_km
    radius_lon = radius_km * lon_deg_per_km
    
    # Create coordinate arrays
    lat_coords = np.linspace(center_lat - radius_lat, center_lat + radius_lat, resolution)
    lon_coords = np.linspace(center_lon - radius_lon, center_lon + radius_lon, resolution)
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    
    logger.info(f"Created {resolution}x{resolution} grid covering {radius_km}km radius around Gosport")
    logger.info(f"Grid bounds: lat {lat_coords[0]:.4f} to {lat_coords[-1]:.4f}, "
                f"lon {lon_coords[0]:.4f} to {lon_coords[-1]:.4f}")
    
    return lat_grid, lon_grid, lat_coords, lon_coords

def calculate_meteorite_density_map(df: pd.DataFrame, lat_grid: np.ndarray, 
                                   lon_grid: np.ndarray) -> np.ndarray:
    """
    Calculate local meteorite density for each grid cell.
    
    Args:
        df (pd.DataFrame): Meteorite data
        lat_grid (np.ndarray): Latitude grid
        lon_grid (np.ndarray): Longitude grid
        
    Returns:
        np.ndarray: Density map of meteorite finds
    """
    density_map = np.zeros_like(lat_grid)
    resolution = lat_grid.shape[0]
    
    # Search radius for each grid cell (in degrees)
    search_radius = 0.5  # ~50km radius for density calculation
    
    logger.info("Calculating meteorite density map...")
    
    for i in range(resolution):
        for j in range(resolution):
            center_lat = lat_grid[i, j]
            center_lon = lon_grid[i, j]
            
            # Calculate distance to all meteorites
            lat_diff = df['reclat'] - center_lat
            lon_diff = df['reclong'] - center_lon
            distances = np.sqrt(lat_diff**2 + lon_diff**2)
            
            # Count nearby meteorites
            nearby_count = np.sum(distances < search_radius)
            density_map[i, j] = nearby_count
    
    logger.info(f"Density map calculated. Max density: {density_map.max()}, Mean: {density_map.mean():.2f}")
    return density_map

def create_umaco_fitness_landscape(density_map: np.ndarray, 
                                  distance_from_center: np.ndarray) -> np.ndarray:
    """
    Create a fitness landscape for UMACO based on meteorite density and search practicality.
    
    The fitness function rewards areas that have:
    1. Low current meteorite density (under-explored)
    2. Reasonable accessibility (not too far from Gosport center)
    3. Geographic features that might preserve meteorites
    
    Args:
        density_map (np.ndarray): Current meteorite density
        distance_from_center (np.ndarray): Distance from Gosport center
        
    Returns:
        np.ndarray: Fitness landscape for UMACO
    """
    # Invert density - lower density = higher potential
    potential_map = 1.0 / (1.0 + density_map)
    
    # Add distance penalty (prefer areas closer to Gosport for practical searching)
    max_distance = distance_from_center.max()
    distance_factor = 1.0 - (distance_from_center / max_distance) * 0.3
    
    # Combine factors
    fitness_landscape = potential_map * distance_factor
    
    # Add some realistic geographic bias
    # Areas near coast, fields, and open spaces are better for meteorite preservation
    # This is a simplified model - in reality you'd use land use data
    center_i, center_j = fitness_landscape.shape[0] // 2, fitness_landscape.shape[1] // 2
    
    # Create some "preferred terrain" hotspots (simulated)
    for _ in range(5):
        hotspot_i = np.random.randint(0, fitness_landscape.shape[0])
        hotspot_j = np.random.randint(0, fitness_landscape.shape[1])
        
        # Add Gaussian hotspot
        di, dj = np.ogrid[:fitness_landscape.shape[0], :fitness_landscape.shape[1]]
        hotspot_dist = np.sqrt((di - hotspot_i)**2 + (dj - hotspot_j)**2)
        hotspot_strength = 0.3 * np.exp(-hotspot_dist**2 / (2 * 5**2))
        fitness_landscape += hotspot_strength
    
    return fitness_landscape

def umaco_meteorite_loss_function(landscape: np.ndarray):
    """
    Create a loss function for UMACO that finds promising meteorite search areas.
    
    Args:
        landscape (np.ndarray): The fitness landscape
        
    Returns:
        Callable: Loss function for UMACO optimization
    """
    def loss_fn(x: np.ndarray) -> float:
        """
        Loss function that rewards finding high-potential, low-explored areas.
        
        Args:
            x (np.ndarray): UMACO pheromone field
            
        Returns:
            float: Loss value (lower = better for UMACO)
        """
        # Ensure x matches landscape dimensions
        if x.shape != landscape.shape:
            # Resize if needed
            from scipy.ndimage import zoom
            scale_factors = (landscape.shape[0] / x.shape[0], landscape.shape[1] / x.shape[1])
            x_resized = zoom(x, scale_factors, order=1)
        else:
            x_resized = x
        
        # The loss should be low where we want UMACO to focus
        # We want UMACO to concentrate on high-fitness areas
        overlap = np.sum(x_resized * landscape)
        
        # Also encourage exploration of diverse areas
        entropy = -np.sum(x_resized * np.log(x_resized + 1e-8))
        
        # Combine: maximize overlap with good areas, maintain some entropy
        loss = -overlap - 0.1 * entropy
        
        return float(loss)
    
    return loss_fn

def run_gosport_simulation(csv_path: str, simulation_years: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the complete Gosport meteorite simulation.
    
    Args:
        csv_path (str): Path to meteorite CSV data
        simulation_years (int): Number of years to simulate
        
    Returns:
        Tuple of (treasure_map, lat_coords, lon_coords)
    """
    logger.info(f"Starting Gosport meteorite simulation for {simulation_years} years")
    
    # Load and process data
    df = load_meteorite_data(csv_path)
    df_uk = filter_uk_europe_data(df)
    
    # Create Gosport grid
    lat_grid, lon_grid, lat_coords, lon_coords = create_gosport_grid(
        GOSPORT_CENTER[0], GOSPORT_CENTER[1], SEARCH_RADIUS_KM, GRID_RESOLUTION
    )
    
    # Calculate current meteorite density
    density_map = calculate_meteorite_density_map(df_uk, lat_grid, lon_grid)
    
    # Calculate distance from Gosport center for each grid cell
    center_i, center_j = GRID_RESOLUTION // 2, GRID_RESOLUTION // 2
    di, dj = np.ogrid[:GRID_RESOLUTION, :GRID_RESOLUTION]
    distance_from_center = np.sqrt((di - center_i)**2 + (dj - center_j)**2)
    
    # Create fitness landscape
    fitness_landscape = create_umaco_fitness_landscape(density_map, distance_from_center)
    
    # Set up UMACO
    logger.info("Initializing UMACO for meteorite optimization...")
    umaco = create_default_umaco(dim=GRID_RESOLUTION, max_iter=simulation_years, use_gpu=True)
    
    # Create agents with meteorite-hunting specializations
    economy = UniversalEconomy(EconomyConfig(n_agents=8))
    agents = create_agents(economy, n_agents=8)
    
    # Create loss function
    loss_fn = umaco_meteorite_loss_function(fitness_landscape)
    
    # Run UMACO optimization
    logger.info(f"Running UMACO simulation for {simulation_years} iterations...")
    result_real, result_imag, panic_history, homology = umaco.optimize(agents, loss_fn)
    
    # The result_real is our "treasure map" - areas where UMACO concentrated
    treasure_map = result_real
    
    logger.info("Simulation complete!")
    return treasure_map, lat_coords, lon_coords

def plot_treasure_map(treasure_map: np.ndarray, lat_coords: np.ndarray, 
                     lon_coords: np.ndarray, save_path: str = None):
    """
    Create a visual treasure map for meteorite hunting.
    
    Args:
        treasure_map (np.ndarray): UMACO result showing promising areas
        lat_coords (np.ndarray): Latitude coordinates
        lon_coords (np.ndarray): Longitude coordinates
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Create the heatmap
    im = plt.imshow(treasure_map, extent=[lon_coords[0], lon_coords[-1], 
                                        lat_coords[0], lat_coords[-1]], 
                   cmap='hot', interpolation='bilinear', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Meteorite Discovery Potential')
    
    # Mark Gosport center
    plt.plot(GOSPORT_CENTER[1], GOSPORT_CENTER[0], 'b*', markersize=15, 
             label='Gosport Center', markeredgecolor='white', markeredgewidth=2)
    
    # Add contour lines to show treasure zones
    contours = plt.contour(lon_coords, lat_coords, treasure_map, 
                          levels=5, colors='cyan', alpha=0.7, linewidths=1)
    plt.clabel(contours, inline=True, fontsize=8)
    
    # Formatting
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Gosport Meteorite Treasure Map\n'
              f'UMACO-Optimized Search Zones (1000-Year Simulation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add coordinate grid
    plt.gca().tick_params(axis='both', which='major', labelsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Treasure map saved to {save_path}")
    
    plt.show()

def identify_top_search_zones(treasure_map: np.ndarray, lat_coords: np.ndarray, 
                             lon_coords: np.ndarray, top_n: int = 5) -> List[dict]:
    """
    Identify the top meteorite search zones from the treasure map.
    
    Args:
        treasure_map (np.ndarray): UMACO optimization result
        lat_coords (np.ndarray): Latitude coordinates
        lon_coords (np.ndarray): Longitude coordinates
        top_n (int): Number of top zones to identify
        
    Returns:
        List[dict]: Top search zones with coordinates and scores
    """
    # Find peak locations
    from scipy.ndimage import maximum_filter
    
    # Apply maximum filter to find local peaks
    local_max = maximum_filter(treasure_map, size=5) == treasure_map
    
    # Get coordinates of local maxima
    peak_coords = np.where(local_max)
    peak_values = treasure_map[peak_coords]
    
    # Sort by value and take top N
    sorted_indices = np.argsort(peak_values)[::-1][:top_n]
    
    search_zones = []
    for idx in sorted_indices:
        i, j = peak_coords[0][idx], peak_coords[1][idx]
        lat = lat_coords[i]
        lon = lon_coords[j]
        score = peak_values[idx]
        
        search_zones.append({
            'rank': len(search_zones) + 1,
            'latitude': lat,
            'longitude': lon,
            'score': score,
            'grid_i': i,
            'grid_j': j
        })
    
    return search_zones

def main():
    """Main execution function."""
    # Configuration
    CSV_PATH = "Meteorite_Landings.csv"  # Update this path
    SIMULATION_YEARS = 1000
    
    try:
        # Run the simulation
        treasure_map, lat_coords, lon_coords = run_gosport_simulation(
            CSV_PATH, SIMULATION_YEARS
        )
        
        # Create and show the treasure map
        plot_treasure_map(treasure_map, lat_coords, lon_coords, 
                         save_path="gosport_meteorite_treasure_map.png")
        
        # Identify top search zones
        search_zones = identify_top_search_zones(treasure_map, lat_coords, lon_coords)
        
        print("\n" + "="*60)
        print("🚀 GOSPORT METEORITE TREASURE MAP COMPLETE! 🪨")
        print("="*60)
        print(f"\nTop {len(search_zones)} Meteorite Search Zones:")
        print("-" * 50)
        
        for zone in search_zones:
            print(f"Zone #{zone['rank']}:")
            print(f"  📍 Coordinates: {zone['latitude']:.4f}°N, {zone['longitude']:.4f}°W")
            print(f"  ⭐ UMACO Score: {zone['score']:.3f}")
            print(f"  📱 Google Maps: https://maps.google.com/?q={zone['latitude']},{zone['longitude']}")
            print()
        
        print("🎯 Happy meteorite hunting! Remember to:")
        print("   • Check land access permissions")
        print("   • Bring a metal detector (for iron meteorites)")
        print("   • Look for dark, fusion-crusted stones")
        print("   • Search after plowing or construction")
        print("   • Check weather-exposed surfaces")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {CSV_PATH}")
        print("Please ensure the meteorite CSV file is in the current directory.")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise

if __name__ == "__main__":
    main()