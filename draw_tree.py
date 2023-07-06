from igraph import Graph
import plotly.graph_objects as go
import plotly.io as io

def draw_tree(tree,title=""): 
    '''
    Plot context tree in a variable length Markov chain in PNG


    Args:
    -----
    tree: Tree object
    
    title: String

    Returns:
    -----
    None: NoneType
    '''
    
    io.renderers.default='png'
    nodes = tree.edges
    nr_vertices =len(nodes) * len(tree.s) + 1
    v_label = [""]
    lab = [0]
    cont = 0
    for children in nodes.values():
        for child in children:
            cont+=1
            v_label += [child]
            lab += [cont]
    
        
    
    G = Graph(directed=True)
    
    
    # Add vertices
    G.add_vertices(nr_vertices)  
    
    # Add edges to create the tree structure
    for father in nodes:
        for children in nodes[father]:
            G.add_edges([(lab[v_label.index(father)],lab[v_label.index(children)])])
    lay = G.layout('rt')
    
    
    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)
    
    E = [e.tuple for e in G.es] # list of edges
    
    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]
    
    labels = v_label
    
    
    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                       y=Ye,
                       mode='lines',
                       line=dict(color='rgb(0,0,0)', width=1),
                       hoverinfo='none'
                       ))
    fig.add_trace(go.Scatter(x=Xn,
                      y=Yn,
                      mode='markers',
                      name='bla',
                      marker=dict(symbol='circle-dot',
                                    size=5,
                                    color='rgb(250,50,50)',    #'#DB4551',
                                    line=dict(color='rgb(250,50,50)', width=1)
                                    ),
                      text=labels,
                      hoverinfo='text',
                      opacity=0.8
                      ))
    def make_annotations(pos, text, font_size=15, font_color='rgb(0,0,0)'):
        L=len(pos)
        if len(text)!=L:
            raise ValueError('The lists pos and text must have the same len')
        annotations = []
        for k in range(L):
            annotations.append(
                dict(
                    text=labels[k], # or replace labels with a different list for the text within the circle
                    x=pos[k][0], y=2*M-position[k][1]-0.2,
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return annotations
    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )
    
    fig.update_layout(title= title,
                  annotations=make_annotations(position, v_label),
                  font_size=12,
                  showlegend=False,
                  xaxis=axis,
                  yaxis=axis,
                  margin=dict(l=40, r=40, b=85, t=100),
                  hovermode='closest',
                  plot_bgcolor='rgb(255,255,255)'
                  )
    fig.show()
