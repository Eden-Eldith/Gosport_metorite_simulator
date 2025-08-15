#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gosport Meteorite Map Visualizer
================================

Overlays UMACO treasure map results on real Gosport geography.
Supports multiple visualization methods depending on available libraries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Gosport area bounds
GOSPORT_CENTER = (50.7963, -1.1267)
GOSPORT_BOUNDS = {
    'north': 50.93,
    'south': 50.66,
    'east': -0.91,
    'west': -1.34
}

def create_map_with_basemap(treasure_map, lat_coords, lon_coords, meteorite_df=None):
    """
    Create map using Basemap (if available).
    """
    try:
        from mpl_toolkits.basemap import Basemap
        
        fig = plt.figure(figsize=(14, 12))
        
        # Create basemap centered on Gosport
        m = Basemap(projection='merc',
                   llcrnrlat=GOSPORT_BOUNDS['south'],
                   urcrnrlat=GOSPORT_BOUNDS['north'],
                   llcrnrlon=GOSPORT_BOUNDS['west'],
                   urcrnrlon=GOSPORT_BOUNDS['east'],
                   resolution='i',  # Intermediate resolution (works without extra install)
                   area_thresh=0.1)
        
        # Draw map features
        m.drawcoastlines(linewidth=1.5, color='darkblue')
        m.drawrivers(linewidth=0.5, color='blue')  # Remove alpha parameter
        m.fillcontinents(color='lightgreen', lake_color='lightblue', alpha=0.3)
        m.drawmapboundary(fill_color='lightblue')
        
        # Draw grid
        parallels = np.arange(50.65, 50.95, 0.05)
        meridians = np.arange(-1.35, -0.90, 0.05)
        m.drawparallels(parallels, labels=[1,0,0,0], fontsize=8, alpha=0.5)
        m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=8, alpha=0.5)
        
        # Create meshgrid for treasure map
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        x, y = m(lon_grid, lat_grid)
        
        # Plot treasure map as overlay
        im = m.pcolormesh(x, y, treasure_map, 
                         cmap='hot', alpha=0.7, 
                         vmin=0, vmax=treasure_map.max())
        
        # Add colorbar
        cbar = plt.colorbar(im, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label('Meteorite Discovery Potential', fontsize=12)
        
        # Mark Gosport center
        gx, gy = m(GOSPORT_CENTER[1], GOSPORT_CENTER[0])
        m.plot(gx, gy, 'c*', markersize=20, 
               markeredgewidth=2, markeredgecolor='white', 
               label='Gosport Center')
        
        # Add key locations
        locations = {
            'Portsmouth': (50.8198, -1.0880),
            'Lee-on-the-Solent': (50.8024, -1.2012),
            'Fareham': (50.8548, -1.1778),
            'Stubbington': (50.8286, -1.2103)
        }
        
        for name, (lat, lon) in locations.items():
            x, y = m(lon, lat)
            m.plot(x, y, 'ko', markersize=5)
            plt.text(x, y, f'  {name}', fontsize=9, ha='left')
        
        # If we have actual meteorite data, plot it
        if meteorite_df is not None:
            uk_meteorites = meteorite_df[
                (meteorite_df['reclat'] >= GOSPORT_BOUNDS['south']) &
                (meteorite_df['reclat'] <= GOSPORT_BOUNDS['north']) &
                (meteorite_df['reclong'] >= GOSPORT_BOUNDS['west']) &
                (meteorite_df['reclong'] <= GOSPORT_BOUNDS['east'])
            ]
            
            if len(uk_meteorites) > 0:
                mx, my = m(uk_meteorites['reclong'].values, uk_meteorites['reclat'].values)
                m.scatter(mx, my, s=50, c='yellow', marker='v', 
                         edgecolors='black', linewidth=1, 
                         label=f'Known Meteorites ({len(uk_meteorites)})', zorder=5)
        
        plt.title('Gosport Area Meteorite Treasure Map\nUMACO-Optimized Search Zones', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', fontsize=10)
        
        return fig
        
    except ImportError:
        print("⚠️ Basemap not available. Using alternative visualization.")
        return None


def create_map_with_cartopy(treasure_map, lat_coords, lon_coords, meteorite_df=None):
    """
    Create map using Cartopy (more modern alternative to Basemap).
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        fig = plt.figure(figsize=(14, 12))
        
        # Create map with Mercator projection
        ax = plt.axes(projection=ccrs.Mercator())
        ax.set_extent([GOSPORT_BOUNDS['west'], GOSPORT_BOUNDS['east'],
                      GOSPORT_BOUNDS['south'], GOSPORT_BOUNDS['north']],
                     crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
        ax.add_feature(cfeature.LAND, color='lightgreen', alpha=0.3)
        ax.add_feature(cfeature.RIVERS, alpha=0.5)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        
        # Plot treasure map
        im = ax.pcolormesh(lon_grid, lat_grid, treasure_map,
                          transform=ccrs.PlateCarree(),
                          cmap='hot', alpha=0.7,
                          vmin=0, vmax=treasure_map.max())
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label('Meteorite Discovery Potential', fontsize=12)
        
        # Mark Gosport
        ax.plot(GOSPORT_CENTER[1], GOSPORT_CENTER[0], 'c*',
               markersize=20, markeredgewidth=2, markeredgecolor='white',
               transform=ccrs.PlateCarree(), label='Gosport Center')
        
        # Add locations
        locations = {
            'Portsmouth': (50.8198, -1.0880),
            'Lee-on-the-Solent': (50.8024, -1.2012),
            'Fareham': (50.8548, -1.1778),
            'Stubbington': (50.8286, -1.2103)
        }
        
        for name, (lat, lon) in locations.items():
            ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree())
            ax.text(lon, lat, f'  {name}', fontsize=9, 
                   transform=ccrs.PlateCarree())
        
        plt.title('Gosport Area Meteorite Treasure Map\nUMACO-Optimized Search Zones',
                 fontsize=16, fontweight='bold')
        plt.legend(loc='upper right')
        
        return fig
        
    except ImportError:
        print("⚠️ Cartopy not available. Using simple visualization.")
        return None


def create_interactive_folium_map(treasure_map, lat_coords, lon_coords, save_path='gosport_meteorite_map.html'):
    """
    Create an interactive HTML map using Folium.
    """
    try:
        import folium
        from folium import plugins
        
        # Create base map
        m = folium.Map(location=GOSPORT_CENTER, 
                      zoom_start=11,
                      tiles='OpenStreetMap')
        
        # Add different tile layers
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        
        # Convert treasure map to heatmap data
        heat_data = []
        for i in range(len(lat_coords)):
            for j in range(len(lon_coords)):
                if treasure_map[i, j] > treasure_map.mean():
                    heat_data.append([lat_coords[i], lon_coords[j], 
                                    float(treasure_map[i, j])])
        
        # Add heatmap
        plugins.HeatMap(heat_data, 
                       min_opacity=0.2,
                       radius=15,
                       blur=10,
                       gradient={0.4: 'blue', 0.6: 'lime', 
                                0.8: 'orange', 1.0: 'red'}).add_to(m)
        
        # Mark Gosport center
        folium.Marker(
            location=GOSPORT_CENTER,
            popup='Gosport Center',
            icon=folium.Icon(color='blue', icon='star')
        ).add_to(m)
        
        # Add search zones as circles
        top_zones = find_top_zones_coords(treasure_map, lat_coords, lon_coords, n=5)
        for i, (lat, lon, score) in enumerate(top_zones, 1):
            folium.Circle(
                location=[lat, lon],
                radius=500,  # 500 meters
                popup=f'Zone #{i}<br>Score: {score:.3f}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}',
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.3
            ).add_to(m)
            
            folium.Marker(
                location=[lat, lon],
                popup=f'Search Zone #{i}',
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        # Add measurement tool
        plugins.MeasureControl().add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(save_path)
        print(f"💾 Interactive map saved: {save_path}")
        print(f"   Open in browser: file://{save_path}")
        
        return m
        
    except ImportError:
        print("⚠️ Folium not available. Install with: pip install folium")
        return None


def create_simple_geographic_map(treasure_map, lat_coords, lon_coords):
    """
    Simple matplotlib visualization with geographic context.
    This should always work as it only uses matplotlib.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    except:
        # Fallback to single plot if subplots fail
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
        ax2 = None
    
    # Left panel: Treasure map with geographic overlay
    im1 = ax1.imshow(treasure_map, 
                     extent=[lon_coords[0], lon_coords[-1], 
                            lat_coords[0], lat_coords[-1]],
                     origin='lower', cmap='hot', aspect='auto', alpha=0.8)
    
    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Mark Gosport and nearby towns
    ax1.plot(GOSPORT_CENTER[1], GOSPORT_CENTER[0], 'c*', 
            markersize=20, markeredgewidth=2, markeredgecolor='white')
    
    locations = {
        'Portsmouth': (50.8198, -1.0880),
        'Lee-on-Solent': (50.8024, -1.2012),
        'Fareham': (50.8548, -1.1778),
        'Stubbington': (50.8286, -1.2103),
        'Gosport': GOSPORT_CENTER
    }
    
    for name, (lat, lon) in locations.items():
        if name != 'Gosport':
            ax1.plot(lon, lat, 'ko', markersize=6)
        ax1.text(lon, lat + 0.01, name, fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add coastline approximation (Solent)
    solent_lons = [-1.30, -1.20, -1.10, -1.00, -0.95]
    solent_lats = [50.72, 50.71, 50.70, 50.72, 50.73]
    ax1.plot(solent_lons, solent_lats, 'b-', linewidth=2, alpha=0.5, label='Solent')
    
    # If single panel, also add top zones here
    if ax2 is None:
        top_zones = find_top_zones_coords(treasure_map, lat_coords, lon_coords, n=5)
        for i, (lat, lon, score) in enumerate(top_zones, 1):
            ax1.plot(lon, lat, 'r*', markersize=15)
            ax1.text(lon, lat - 0.01, f'Zone #{i}', fontsize=9, ha='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Meteorite Treasure Map - Geographic View')
    ax1.legend(loc='upper right')
    
    plt.colorbar(im1, ax=ax1, label='Discovery Potential')
    
    # Right panel: Contour plot with zones (if we have two panels)
    if ax2 is not None:
        contours = ax2.contour(lon_coords, lat_coords, treasure_map, 
                              levels=10, cmap='viridis')
        ax2.clabel(contours, inline=True, fontsize=8)
        
        im2 = ax2.contourf(lon_coords, lat_coords, treasure_map, 
                           levels=20, cmap='YlOrRd', alpha=0.6)
        
        # Mark top zones
        top_zones = find_top_zones_coords(treasure_map, lat_coords, lon_coords, n=5)
        for i, (lat, lon, score) in enumerate(top_zones, 1):
            ax2.plot(lon, lat, 'r*', markersize=15)
            ax2.text(lon, lat + 0.005, f'#{i}', fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Top Search Zones - Contour View')
        ax2.grid(True, alpha=0.3)
        
        plt.colorbar(im2, ax=ax2, label='Discovery Potential')
    
    if ax2 is not None:
        plt.suptitle('Gosport Meteorite Hunting Map - UMACO Analysis', 
                    fontsize=16, fontweight='bold')
    else:
        ax1.set_title('Gosport Meteorite Hunting Map - UMACO Analysis', 
                     fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def find_top_zones_coords(treasure_map, lat_coords, lon_coords, n=5):
    """
    Find top n search zones with their coordinates.
    """
    zones = []
    temp_map = treasure_map.copy()
    
    for _ in range(n):
        # Find peak
        peak_idx = np.unravel_index(np.argmax(temp_map), temp_map.shape)
        i, j = peak_idx
        
        lat = lat_coords[i]
        lon = lon_coords[j]
        score = treasure_map[i, j]
        
        zones.append((lat, lon, score))
        
        # Zero out area around peak
        for di in range(max(0, i-3), min(temp_map.shape[0], i+4)):
            for dj in range(max(0, j-3), min(temp_map.shape[1], j+4)):
                temp_map[di, dj] = 0
    
    return zones


def visualize_treasure_map(treasure_map, lat_coords, lon_coords, 
                          meteorite_df=None, output_prefix='gosport'):
    """
    Main function to create all available visualizations.
    """
    print("\n" + "="*60)
    print("📍 CREATING GEOGRAPHIC VISUALIZATIONS")
    print("="*60)
    
    fig = None
    
    # Try Basemap first (most detailed)
    try:
        fig = create_map_with_basemap(treasure_map, lat_coords, lon_coords, meteorite_df)
        if fig:
            plt.savefig(f'{output_prefix}_basemap.png', dpi=200, bbox_inches='tight')
            print(f"✅ Basemap visualization saved: {output_prefix}_basemap.png")
            plt.show()
    except Exception as e:
        print(f"⚠️ Basemap failed: {e}")
        fig = None
    
    # Try Cartopy (modern alternative)
    if not fig:
        try:
            fig = create_map_with_cartopy(treasure_map, lat_coords, lon_coords, meteorite_df)
            if fig:
                plt.savefig(f'{output_prefix}_cartopy.png', dpi=200, bbox_inches='tight')
                print(f"✅ Cartopy visualization saved: {output_prefix}_cartopy.png")
                plt.show()
        except Exception as e:
            print(f"⚠️ Cartopy failed: {e}")
            fig = None
    
    # Always create simple geographic map (works with just matplotlib)
    if not fig:
        try:
            fig = create_simple_geographic_map(treasure_map, lat_coords, lon_coords)
            plt.savefig(f'{output_prefix}_geographic.png', dpi=200, bbox_inches='tight')
            print(f"✅ Geographic visualization saved: {output_prefix}_geographic.png")
            plt.show()
        except Exception as e:
            print(f"❌ Even simple visualization failed: {e}")
    
    # Try creating interactive map
    try:
        create_interactive_folium_map(treasure_map, lat_coords, lon_coords, 
                                     f'{output_prefix}_interactive.html')
    except Exception as e:
        print(f"⚠️ Interactive map failed: {e}")
    
    # Print top zones with coordinates
    print("\n" + "-"*60)
    print("🎯 TOP METEORITE SEARCH ZONES (with GPS coordinates):")
    print("-"*60)
    
    top_zones = find_top_zones_coords(treasure_map, lat_coords, lon_coords, n=5)
    for i, (lat, lon, score) in enumerate(top_zones, 1):
        print(f"\n📍 Zone #{i}:")
        print(f"   GPS: {lat:.5f}°N, {abs(lon):.5f}°W")
        print(f"   Score: {score:.1%}")
        print(f"   Google Maps: https://maps.google.com/?q={lat},{lon}")
        print(f"   What3Words: https://what3words.com/{lat},{lon}")
        
        # Estimate nearest landmark
        min_dist = float('inf')
        nearest = ''
        locations = {
            'Gosport Town': GOSPORT_CENTER,
            'Portsmouth': (50.8198, -1.0880),
            'Lee-on-Solent': (50.8024, -1.2012),
            'Fareham': (50.8548, -1.1778),
            'Stubbington': (50.8286, -1.2103)
        }
        
        for name, (loc_lat, loc_lon) in locations.items():
            dist = np.sqrt((lat - loc_lat)**2 + (lon - loc_lon)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = name
                
        print(f"   Nearest: {nearest} (~{min_dist*111:.1f} km)")
    
    print("\n" + "="*60)
    print("✅ All visualizations complete!")
    print("="*60)


# Example usage with your UMACO results
if __name__ == "__main__":
    # Example: Load your UMACO results
    # This would come from your pure_gpu_umaco.py output
    
    # Create sample data for testing
    lat_coords = np.linspace(GOSPORT_BOUNDS['south'], GOSPORT_BOUNDS['north'], 64)
    lon_coords = np.linspace(GOSPORT_BOUNDS['west'], GOSPORT_BOUNDS['east'], 64)
    
    # Sample treasure map (replace with your actual UMACO output)
    treasure_map = np.random.rand(64, 64)
    
    # Add some hotspots for visualization
    for _ in range(5):
        i, j = np.random.randint(0, 64, 2)
        treasure_map[max(0,i-2):min(64,i+3), max(0,j-2):min(64,j+3)] += 0.5
    
    treasure_map = (treasure_map - treasure_map.min()) / (treasure_map.max() - treasure_map.min())
    
    # Try loading meteorite data if available
    try:
        meteorite_df = pd.read_csv('Meteorite_Landings.csv')
        meteorite_df = meteorite_df.dropna(subset=['reclat', 'reclong'])
        print(f"📊 Loaded {len(meteorite_df)} meteorite records")
    except:
        meteorite_df = None
        print("⚠️ Meteorite data not loaded")
    
    # Create visualizations
    visualize_treasure_map(treasure_map, lat_coords, lon_coords, 
                          meteorite_df, output_prefix='gosport_test')