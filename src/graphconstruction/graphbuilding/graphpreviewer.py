from typing import Literal, assert_never, Sequence, cast, Any
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import geopandas as gpd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ...utils.types import AdminLevel, Country
from ..graphobjects import GraphStructure

class GraphViewer:
    """ 
    Class to preview GraphStructures with.

    Parameters
    ----------
    background_shapedata : gpd.GeoDataFrame
        Shapedata for the entire country at all admin levels.
    level_shapedata : gpd.GeoDataFrame
        Shapedata of tokenized admin level.
    country : Country
        Country studied.
    level : AdminLevel
        Admin level studied.

    Downstream
    ----------
    ``GraphManager`` manages the creation of Graph Objects. It also allows one to preview graphs,
    since its ``preview()`` function interacts with ``GraphViewer``. Each ``GraphManager`` has a
    single ``GraphViewer`` class associated with attribute ``previewer``.
    """
    level: AdminLevel
    country: Country
    def __init__(self,
                 background_shapedata:  gpd.GeoDataFrame,
                 level_shapedata:       gpd.GeoDataFrame,
                 country:               Country,
                 level:                 AdminLevel):
        
        self.background_shapedata   = background_shapedata
        self.level_shapedata        = level_shapedata
        self.level                  = level
        self.country                = country

        self._set_style_defaults()

    # ======== METHODS ========= #
    def view(self, 
             graph_structure:   GraphStructure, 
             variable:          Literal['network','degree'],
             locality:          Literal['local','global'],
             plot_type:         Literal['histogram','map'],
             neighborhood:      int | None,
             connections_type:  Literal['in','out'] | None,
             *args, **kwargs) -> tuple[Figure, Axes]:    
        """        
        Main function of GraphViewer.

        While this function may also be called through the ``GraphManager`` using its
        ``preview()`` function, using the ``Graphviewer`` by itself allows one to preview
        any GraphStructe (not GraphObject!).

        Examples
        --------
        >>> pm = PathManager()
        ... graph3 = GraphObject.load(pm.data / 'germany/graphs/nuts3/graph3/')
        ... gm.previewer.view(graph3.graph,'degree','global','map', None,None)
        """
        fig:    Figure
        ax:     Axes
        axes:   Sequence[Axes]

        # If necessary, filter on neighborhood (with node - int and connections_type)
        if locality == 'local' and isinstance(neighborhood, int) and connections_type is not None:
            index_to_plot, weight_to_plot = self._filter_neighborhood(graph_structure, neighborhood, connections_type)
        else:
            index_to_plot = graph_structure.edge_index
            weight_to_plot = graph_structure.edge_weight

        match plot_type:
            case 'histogram':
                fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            case 'map':
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))            
                self._plot_background_map_on_ax(ax)            
            case _:
                assert_never(plot_type)

        # decide on plotting - function
        match (variable, plot_type):

            case ('degree', 'histogram'):    
                self._plot_degree_histogram_on_ax(index_to_plot, 
                                                  ax, 
                                                  connections_type)

                title = f"Degree distribution"                       

            case ('network', 'map'):
                self._plot_network_lines_on_ax(index_to_plot, 
                                               locality, 
                                               ax)

                self._plot_geompoints_on_ax(neighborhood = neighborhood, 
                                            node_geoms = True, 
                                            neighbor_geoms = True, 
                                            isolate_geoms = True, 
                                            edge_index = index_to_plot, 
                                            ax = ax)
                
                self._update_map_layout(neighborhood = neighborhood, 
                                        node_geoms = True, 
                                        neighbor_geoms = True, 
                                        isolate_geoms = True, 
                                        ax = ax)                

                title = f"Network"

            case ('degree', 'map'):
                self._plot_degree_map_on_ax(index_to_plot, 
                                            connections_type, 
                                            ax, 
                                            fig)
                
                self._update_map_layout(neighborhood = neighborhood, 
                                        node_geoms = False, 
                                        neighbor_geoms = False, 
                                        isolate_geoms = False, 
                                        ax = ax)

                title = f"Degree distribution"

            case _:
                raise NotImplementedError(
                    f'No implementation for: {(variable, locality, plot_type, neighborhood, connections_type)}'
                    )


        locality_part_of_title = f" - {locality}"
        
        if neighborhood is not None:
            locality_part_of_title += f' - node {neighborhood}'

        if connections_type is not None:
            locality_part_of_title += f' - {connections_type.upper()}'

        title = title + locality_part_of_title

        self._update_title(title = title, ax = ax)

        return fig, ax

    # ====== HIDDEN METHODS ========= #

    # ===== HELPERS

    # filtering on neighborhood
    def _filter_neighborhood(self, 
                             graph_structure: GraphStructure, 
                             neighborhood_node: int, 
                             connections_type: Literal['in','out']) -> tuple[torch.Tensor, torch.Tensor]:
        """Given neighborhood integer and type of connections, filters the specific neighborhood"""
        node_out, node_in = graph_structure.edge_index  # unpack tensors

        match connections_type:
            case 'in':
                mask = node_in == neighborhood_node
            case 'out':
                mask = node_out == neighborhood_node
            case _:
                assert_never(connections_type)

        # edges_of_interest = indices of edges that match
        edges_of_interest   = mask.nonzero(as_tuple=True)[0]
        selected_edges      = graph_structure.edge_index[:, edges_of_interest]  # [2, num_selected]
        selected_weights    = graph_structure.edge_weight[edges_of_interest]

        return selected_edges, selected_weights

    # level - to exclude from background map
    def _get_level_to_exclude(self) -> list[str]:
        """validate input combinations"""
        level           = self.level
        country         = self.country

        match (country, level):

            case ('germany','nuts3'):
                exclude_levels = ['nuts3']
            case ('germany','nuts2'):
                exclude_levels = ['nuts3','nuts2']        
            case ('germany','nuts1'):
                exclude_levels = ['nuts3','nuts2','nuts1']        

            case ('hungary','nuts3'):
                exclude_levels = ['nuts3']
            case ('hungary','nuts2'):
                exclude_levels = ['nuts3','nuts2']        
            case ('hungary','nuts1'):
                exclude_levels = ['nuts3','nuts2','nuts1']    

            case _:
                exclude_levels = []
            
        return exclude_levels

    # compute - in / out degree
    def _compute_degree(self, 
                        edge_index: torch.Tensor, 
                        connections_type: Literal['in', 'out'] | None = None) -> torch.Tensor:
        """
        Computes per-node degree from edge_index 
        
        Parameters
        ----------
        edge_index: torch.Tensor

        connections_type: Optional[Literal['in','out']] = None
            - 'in':  count incoming edges per node
            - 'out': count outgoing edges per node
            - None:  sum of both
        """
        num_nodes = int(edge_index.max().item() + 1)
        ones      = torch.ones(edge_index.shape[1], dtype=torch.int)
        degree    = torch.zeros(num_nodes, dtype=torch.int)

        match connections_type:
            case 'in':
                degree.scatter_add_(0, edge_index[1], ones)
            case 'out':
                degree.scatter_add_(0, edge_index[0], ones)
            case None:
                degree.scatter_add_(0, edge_index[0], ones)
                degree.scatter_add_(0, edge_index[1], ones)
            case _:
                assert_never(connections_type)

        return degree
 
    # ==== PLOTTERS

    def _update_title(self, title: str, ax: Axes):
        ax.set_title(title)        

    def _set_style_defaults(self):

        self.map_level_patch_styles: dict[str, dict[str, Any]]= {
                'nuts3' :       dict(facecolor='none',      edgecolor='darkgrey',linewidth=0.2, zorder = 2),
                'nuts2' :       dict(facecolor='none',      edgecolor='black',  linewidth = 0.3,zorder = 3),        
                'nuts1' :       dict(facecolor='none',      edgecolor='black',  linewidth = 1,  zorder = 4),        
                'nuts0' :       dict(facecolor='none',      edgecolor='black',  linewidth = 2,  zorder = 5),        
        }    

        self.node_class_styles: dict[str, dict[str, Any]] = {
            'node'      :   dict(markersize=75, edgecolor='black', color='red',                 zorder = 7),
            'neighbor'  :   dict(markersize=45, edgecolor='black', color='orange',              zorder = 8),
            'isolate'   :   dict(markersize=25, edgecolor='black', color='lightgrey',           zorder = 9)
        }

        self.node_class_styles_legend: dict[str, dict[str, Any]] = {
            'node'      :   dict(marker = 'o', markersize=10, markeredgecolor='black', label = 'node',      markerfacecolor='red',      linestyle = 'none'),
            'neighbor'  :   dict(marker = 'o', markersize=10, markeredgecolor='black', label = 'neighbor',  markerfacecolor='orange',   linestyle = 'none'),
            'isolate'   :   dict(marker = 'o', markersize=10, markeredgecolor='black', label = 'isolate',   markerfacecolor='lightgrey',linestyle = 'none')
        }  

    # histograms    
    def _plot_degree_histogram_on_ax(self, edge_index: torch.Tensor, ax: Axes, connections_type: Literal['in', 'out'] | None = None):
        """calling upon `_compute_degree()`, plot histogram of number of connections"""
        degree = self._compute_degree(edge_index, connections_type)
        bins   = np.arange(0, degree.max().item() + 2) - 0.5

        sns.histplot(degree.numpy(), ax=ax, bins=bins)
        ax.set_ylabel('frequency')
        ax.set_xlabel('connections')

    # maps
    def _plot_background_map_on_ax(self, ax: Axes):
        """plots the background map, which is basically al geographical levels, exlcuding the ones found from `_get_level_to_exclude()`"""

        background_shape= self.background_shapedata.copy()
        level_shape     = self.level_shapedata.copy()
        exclude_levels  = self._get_level_to_exclude()
        background_shape= background_shape[~background_shape['level'].isin(exclude_levels)]

        main_level          = self.level
        
        if main_level not in self.map_level_patch_styles:
            raise ValueError(f'didnt find any styles for level {main_level} for country {self.country.lower()}')            

        style   = self.map_level_patch_styles
        main_level_style= style[main_level].copy()
        level_shape.plot(ax= ax, **main_level_style)

        level_shape_points  = level_shape.copy()
        level_shape_points['geometry'] = level_shape_points.representative_point()

        # looping over levels and associated style
        for background_level, level_style in style.items():

            # if this level is inside the shapefile (after exclusion)
            if background_level and background_level in background_shape['level'].unique():

                selection_gdf   = background_shape[background_shape['level'] == background_level]
                selection_gdf.plot(ax=ax, **level_style)              

    def _update_map_layout(self, neighborhood: int | None, node_geoms: bool, neighbor_geoms: bool, isolate_geoms: bool, ax: Axes):
        """fixes the map - layout"""

        # legend
        handles: list[Line2D | Patch] = []     
        style   = self.map_level_patch_styles
        # looping over levels and associated style
        for background_level, level_style in style.items():

            # if this level is inside the shapefile (after exclusion)
            if background_level:
                handles.append(Patch(label = background_level, **level_style))              
            
        if neighborhood is not None and node_geoms:
            handles.append(Line2D([0], [0], **self.node_class_styles_legend['node']))

        if neighbor_geoms:
            handles.append(Line2D([0], [0], **self.node_class_styles_legend['neighbor']))
        
        if isolate_geoms:
            handles.append(Line2D([0], [0], **self.node_class_styles_legend['isolate']))    

        ax.legend(handles=handles, title='', loc='upper left')

        # finally
        ax.set_xticks([])
        ax.set_yticks([])  

    def _plot_geompoints_on_ax(self, neighborhood: int | None, node_geoms: bool, neighbor_geoms: bool, isolate_geoms: bool, edge_index: torch.Tensor, ax: Axes):
        """
        plots geompoints on map, by dividng in 3:
        - node_point (when neighborhood is an integer)
        - connected_points (connected points to node_point if node_point, otherwise nodes that are connected to at least one other node)
        - background_points (points not connected to node_point if node_point, otherwise nodes that are not connected to any other node)
        """
        all_points                  = self.level_shapedata.copy()
        all_points['geometry']      = all_points.representative_point()      

        connected_points       = all_points[all_points['node'].isin(set(edge_index.flatten().tolist()))]
        background_points      = all_points[~all_points['node'].isin(set(edge_index.flatten().tolist()))]

        if neighborhood is not None:
            node_point             = all_points[all_points['node'] == neighborhood]   
            connected_points       = connected_points[connected_points['node']!=neighborhood]
        
            if node_geoms:
                node_point.plot(ax = ax,  **self.node_class_styles['node'])  

        if neighbor_geoms and len(connected_points)>0:                             
            connected_points.plot(ax = ax, **self.node_class_styles['neighbor'])    
        if isolate_geoms and len(background_points)>0:                  
            background_points.plot(ax = ax, **self.node_class_styles['isolate'])     

    def _plot_network_lines_on_ax(self, edge_index: torch.Tensor, locality: Literal['local','global'], ax: Axes):
        """
        plot network lines from given edge_index.

        Parameters
        ----------
        edge_index: torch.Tensor
            edge_index to show connections in. Can already be filtered on neighborhood / connections-type
        locality: Literal['local','global']
            if local, the direction is shown.
        
        NOTE
        ----
        - This map shows qualitative connects. for quantitative looks
          Try `plot_neighborhood_weights_map_on_ax()`.   
        """
        # Get lines
        connection_lines                = self.level_shapedata.copy()
        connection_lines['geometry']    = connection_lines.representative_point()
        connection_lines['x']           = connection_lines.geometry.x
        connection_lines['y']           = connection_lines.geometry.y

        # Build coordinate lookup: node_id (int) -> (x (float), y (float))
        coord_lookup = cast(
            dict[int, dict[str, float]],
            connection_lines.set_index('node')[['x', 'y']].to_dict('index')
        )
            
        # Build coordinate arrays for all edges
        src_nodes = np.array(edge_index[0])
        dst_nodes = np.array(edge_index[1])

        mask = np.isin(src_nodes, list(coord_lookup.keys())) & np.isin(dst_nodes, list(coord_lookup.keys()))
        src_nodes = src_nodes[mask]
        dst_nodes = dst_nodes[mask]

        src_coords = np.array([[coord_lookup[i]['x'], coord_lookup[i]['y']] for i in src_nodes])
        dst_coords = np.array([[coord_lookup[i]['x'], coord_lookup[i]['y']] for i in dst_nodes])

        # Draw all lines at once
        for x0, y0, x1, y1 in zip(src_coords[:,0], src_coords[:,1], dst_coords[:,0], dst_coords[:,1]):
            ax.plot((x0, x1), (y0, y1), color='black', linewidth=0.9, alpha=0.9, zorder=6)
        
        # if local, then also draw arrow heads
        if locality == 'local':
            # fraction along the line for the arrow tip (where the head will be)
            tip_frac = 0.75  

            # fraction behind the tip for the tail of the arrow
            tail_frac = 0.75  # how far back the tail starts (controls arrow length)

            for x0, y0, x1, y1 in zip(src_coords[:,0], src_coords[:,1], dst_coords[:,0], dst_coords[:,1]):
                dx = x1 - x0
                dy = y1 - y0

                # arrow tip at midpoint
                tip_x = x0 + tip_frac * dx
                tip_y = y0 + tip_frac * dy

                # arrow tail a little before tip
                tail_x = x0 + (tip_frac - tail_frac) * dx
                tail_y = y0 + (tip_frac - tail_frac) * dy

                ax.annotate(
                    '',
                    xy=(tip_x, tip_y),      # arrowhead here
                    xytext=(tail_x, tail_y),# tail starts here
                    arrowprops=dict(
                        arrowstyle='-|>',   # normal arrowhead
                        color='black',
                        linewidth=0.9,
                        alpha=0.8
                    ),
                    zorder=6
                )

    def _plot_degree_map_on_ax(self, edge_index: torch.Tensor, connections_type: Literal['in', 'out'] | None, ax: Axes, fig: Figure):
        """plots connections-degree on map. Depending on in/out or None (in+out)"""

        degree = self._compute_degree(edge_index, connections_type)
        cbar_title = 'Degree'

        shape          = self.level_shapedata.copy()
        shape['degree'] = shape['node'].apply(
            lambda n: degree[n].item() if n < len(degree) else 0
        )

        shape.plot(
            column='degree',
            cmap='Blues',
            ax=ax,
            legend=False,
            legend_kwds={'label': cbar_title, 'orientation': 'vertical'},
            edgecolor='darkgrey',
            linewidth=0.3,
            zorder = 1
        )
        mappable = ax.collections[-1] 
        fig.colorbar(mappable, ax=ax, label=cbar_title, shrink=0.5)