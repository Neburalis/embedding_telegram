import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
import plotly.express as px
import pandas as pd
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import os
from main import process_messages
import base64
import io
import threading
import time
import signal
import sys

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Global variable to store processing status
processing_status = {
    'is_processing': False,
    'progress': 0,
    'total': 0,
    'processed': 0,
    'error': None
}

# Define the layout for the file upload page
upload_layout = html.Div([
    html.H1("Upload Messages File", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Label("Embedding Generation Method:"),
            dcc.RadioItems(
                id='embedding-method',
                options=[
                    {'label': 'Generate new embeddings', 'value': 'generate'},
                    {'label': 'Use existing embeddings.json', 'value': 'existing'}
                ],
                value='generate',
                labelStyle={'display': 'inline-block', 'margin': '10px'}
            ),
        ], style={'width': '100%', 'textAlign': 'center', 'margin': '10px'}),
        
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='output-data-upload'),
        html.Div(id='progress-container', style={'display': 'none'}, children=[
            html.Div(id='progress-status'),
            html.Progress(id='progress-bar', value="0", max="100", style={'width': '100%', 'height': '20px'}),
            html.Div(id='progress-details')
        ]),
        dcc.Interval(
            id='progress-interval',
            interval=500,  # Update every 500ms
            n_intervals=0,
            disabled=True
        )
    ], style={'width': '50%', 'margin': '0 auto'})
])

# Define the layout for the visualization page
def create_visualization_layout(embeddings_file: str, messages_file: str):
    # Load and process data
    embeddings, ids, texts, from_fields = load_data(embeddings_file, messages_file)
    embeddings, ids, texts, from_fields = remove_duplicates(embeddings, ids, texts, from_fields)
    
    # Get unique senders for the dropdown
    unique_senders = sorted(list(set(from_fields)))
    
    # Project embeddings to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    return html.Div([
        html.H1("Message Embeddings Visualization", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.Label("Filter by Sender:"),
                dcc.Dropdown(
                    id='sender-filter',
                    options=[{'label': 'All Senders', 'value': 'all'}] + 
                            [{'label': sender, 'value': sender} for sender in unique_senders],
                    value='all',
                    style={'width': '100%', 'margin': '10px'}
                ),
            ], style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.Label("Clustering Algorithm:"),
                dcc.RadioItems(
                    id='algorithm-selector',
                    options=[
                        {'label': 'DBSCAN', 'value': 'DBSCAN'},
                        {'label': 'K-means', 'value': 'K-means'}
                    ],
                    value='DBSCAN',
                    labelStyle={'display': 'inline-block', 'margin': '10px'}
                ),
            ], style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}),
        ], style={'width': '100%', 'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.Label("EPS:"),
                dcc.Slider(
                    id='eps-slider',
                    min=0.1,
                    max=3.0,
                    step=0.1,
                    value=1.6,
                    marks={i/10: str(i/10) for i in range(1, 31, 2)}
                ),
            ], style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.Label("Min Samples:"),
                dcc.Slider(
                    id='min-samples-slider',
                    min=1,
                    max=20,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(1, 21, 2)}
                ),
            ], style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}),
        ], id='dbscan-controls'),
        
        html.Div([
            html.Div([
                html.Label("Number of Clusters:"),
                dcc.Slider(
                    id='n-clusters-slider',
                    min=2,
                    max=200,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in range(2, 201, 20)}
                ),
            ], style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}),
        ], id='kmeans-controls', style={'display': 'none'}),
        
        html.Div(id='clustering-stats', style={'textAlign': 'center', 'margin': '10px'}),
        
        dcc.Graph(
            id='scatter-plot',
            style={'height': '80vh'}
        ),
        
        # Store the processed data
        dcc.Store(id='embeddings-2d', data=embeddings_2d.tolist()),
        dcc.Store(id='ids', data=ids),
        dcc.Store(id='texts', data=texts),
        dcc.Store(id='from-fields', data=from_fields)
    ])

# Define the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

def load_data(embeddings_file: str, messages_file: str):
    """Load embeddings and original messages data."""
    # Load embeddings
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        embeddings_data = json.load(f)
    
    # Load original messages
    with open(messages_file, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)
    
    # Create message_id to text mapping
    message_map = {
        msg['id']: {
            'text': msg['text'],
            'from': msg.get('from', 'Unknown')  # Get 'from' field with default 'Unknown'
        }
        for msg in chat_data['messages']
        if msg['type'] == 'message' and msg['text']
    }
    
    # Prepare data for visualization
    ids = []
    embeddings = []
    texts = []
    from_fields = []
    
    for item in embeddings_data:
        msg_id = item['id']
        if msg_id in message_map:
            ids.append(msg_id)
            embeddings.append(item['embedding'])
            texts.append(message_map[msg_id]['text'])
            from_fields.append(message_map[msg_id]['from'])
    
    return np.array(embeddings), ids, texts, from_fields

def remove_duplicates(embeddings, ids, texts, from_fields):
    """Remove messages with identical embeddings."""
    # Convert embeddings to list of tuples for hashing
    embedding_tuples = [tuple(emb) for emb in embeddings]
    
    # Create a dictionary to store unique embeddings
    unique_data = {}
    for i, emb_tuple in enumerate(embedding_tuples):
        if emb_tuple not in unique_data:
            unique_data[emb_tuple] = {
                'id': ids[i],
                'text': texts[i],
                'from': from_fields[i],
                'embedding': embeddings[i]
            }
    
    # Extract unique data
    unique_embeddings = np.array([data['embedding'] for data in unique_data.values()])
    unique_ids = [data['id'] for data in unique_data.values()]
    unique_texts = [data['text'] for data in unique_data.values()]
    unique_from_fields = [data['from'] for data in unique_data.values()]
    
    return unique_embeddings, unique_ids, unique_texts, unique_from_fields

def cluster_embeddings(embeddings_2d, algorithm, eps, min_samples, n_clusters):
    """Cluster 2D embeddings using selected algorithm."""
    if algorithm == 'DBSCAN':
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings_2d)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        n_clustered = len(labels) - n_noise
    else:  # K-means
        # Perform K-means clustering
        clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings_2d)
        labels = clustering.labels_
        n_clusters = n_clusters
        n_noise = 0
        n_clustered = len(labels)
    
    return labels, n_clusters, n_noise, n_clustered

# Callback for page routing
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/visualize':
        if os.path.exists('embeddings.json') and os.path.exists('result.json'):
            return create_visualization_layout('embeddings.json', 'result.json')
        else:
            return html.Div([
                html.H3("Please upload a file first"),
                html.A("Go to upload page", href="/")
            ])
    else:
        return upload_layout

def process_file_in_background(input_file, output_file):
    """Process file in background thread and update progress."""
    global processing_status
    
    try:
        # Read input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # Get messages array from the chat data
        messages = chat_data.get('messages', [])
        
        # Process each message and get embeddings
        results = []
        total_messages = len(messages)
        processed_count = 0
        
        processing_status['total'] = total_messages
        processing_status['processed'] = 0
        processing_status['is_processing'] = True
        processing_status['error'] = None
        
        for message in messages:
            # Skip if not a message type or if text is empty
            if message.get('type') != 'message' or not message.get('text'):
                continue
                
            message_id = message.get('id')
            message_text = message.get('text', '')
            
            if message_id and message_text:
                try:
                    from main import get_embedding
                    embedding = get_embedding(message_text)
                    results.append({
                        'id': message_id,
                        'embedding': embedding
                    })
                    processed_count += 1
                    processing_status['processed'] = processed_count
                except Exception as e:
                    print(f"Error processing message {message_id}: {str(e)}")
        
        # Save results to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        processing_status['is_processing'] = False
        
    except Exception as e:
        processing_status['error'] = str(e)
        processing_status['is_processing'] = False

# Callback for file upload
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('progress-container', 'style'),
     Output('progress-interval', 'disabled')],
    [Input('upload-data', 'contents'),
     Input('progress-interval', 'n_intervals'),
     Input('embedding-method', 'value')],
    [State('upload-data', 'filename')]
)
def update_output(contents, n_intervals, embedding_method, filename):
    global processing_status
    
    if contents is None:
        return html.Div(), {'display': 'none'}, True
    
    # If this is the first callback (file upload)
    if n_intervals == 0:
        # Save the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        with open('result.json', 'wb') as f:
            f.write(decoded)
        
        if embedding_method == 'generate':
            # Start processing in background
            thread = threading.Thread(
                target=process_file_in_background,
                args=('result.json', 'embeddings.json')
            )
            thread.start()
            return html.Div(), {'display': 'block'}, False
        else:
            # Check if embeddings.json exists
            if os.path.exists('embeddings.json'):
                return (
                    html.Div([
                        html.H3("Using existing embeddings file!"),
                        html.A("Go to visualization", href="/visualize")
                    ]),
                    {'display': 'none'},
                    True
                )
            else:
                return (
                    html.Div([
                        html.H3("Error: embeddings.json not found!"),
                        html.P("Please generate embeddings first or ensure embeddings.json exists in the project directory.")
                    ]),
                    {'display': 'none'},
                    True
                )
    
    # Update progress
    if processing_status['is_processing']:
        progress = (processing_status['processed'] / processing_status['total']) * 100
        return (
            html.Div([
                html.H3("Processing messages..."),
                html.P(f"Processed {processing_status['processed']} of {processing_status['total']} messages")
            ]),
            {'display': 'block'},
            False
        )
    elif processing_status['error']:
        return (
            html.Div([
                html.H3("Error processing file:"),
                html.P(processing_status['error'])
            ]),
            {'display': 'none'},
            True
        )
    else:
        return (
            html.Div([
                html.H3("File processed successfully!"),
                html.A("Go to visualization", href="/visualize")
            ]),
            {'display': 'none'},
            True
        )

# Callback for progress bar
@app.callback(
    [Output('progress-bar', 'value'),
     Output('progress-status', 'children'),
     Output('progress-details', 'children')],
    Input('progress-interval', 'n_intervals')
)
def update_progress(n_intervals):
    if not processing_status['is_processing']:
        return "100", "Complete!", ""
    
    progress = (processing_status['processed'] / processing_status['total']) * 100
    status = f"Processing: {progress:.1f}%"
    details = f"Processed {processing_status['processed']} of {processing_status['total']} messages"
    
    return str(progress), status, details

# Callback for toggling controls
@app.callback(
    [Output('dbscan-controls', 'style'),
     Output('kmeans-controls', 'style')],
    Input('algorithm-selector', 'value')
)
def toggle_controls(algorithm):
    if algorithm == 'DBSCAN':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

# Callback for updating the graph
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('clustering-stats', 'children')],
    [Input('algorithm-selector', 'value'),
     Input('eps-slider', 'value'),
     Input('min-samples-slider', 'value'),
     Input('n-clusters-slider', 'value'),
     Input('embeddings-2d', 'data'),
     Input('ids', 'data'),
     Input('texts', 'data'),
     Input('from-fields', 'data'),
     Input('sender-filter', 'value')]
)
def update_graph(algorithm, eps, min_samples, n_clusters, embeddings_2d, ids, texts, from_fields, selected_sender):
    if not embeddings_2d:
        return {}, html.Div()
    
    embeddings_2d = np.array(embeddings_2d)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'id': ids,
        'text': texts,
        'from': from_fields
    })
    
    # Filter by selected sender if not 'all'
    if selected_sender != 'all':
        df = df[df['from'] == selected_sender]
        embeddings_2d = embeddings_2d[df.index]
    
    # Perform clustering with current parameters
    labels, n_clusters, n_noise, n_clustered = cluster_embeddings(
        embeddings_2d, algorithm, eps, min_samples, n_clusters
    )
    
    df['cluster'] = [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
    
    # Create figure
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        hover_data=['id', 'text', 'from'],
        title='Message Embeddings Visualization with Clusters'
    )
    
    # Update layout
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        title_x=0.5,
        title_font_size=20,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            title=None
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            title=None
        )
    )
    
    # Remove hover coordinates
    fig.update_traces(
        hovertemplate="<b>ID:</b> %{customdata[0]}<br><b>From:</b> %{customdata[2]}<br><b>Text:</b> %{customdata[1]}<extra></extra>"
    )
    
    # Create statistics text
    stats = html.Div([
        html.H3(f"Clustering Statistics:"),
        html.P(f"Algorithm: {algorithm}"),
        html.P(f"Number of clusters: {n_clusters}"),
        html.P(f"Number of noise points: {n_noise}"),
        html.P(f"Number of clustered points: {n_clustered}"),
        html.P(f"Filtered by sender: {selected_sender if selected_sender != 'all' else 'All senders'}")
    ])
    
    return fig, stats

def _shutdown(signum, frame):
    print(f"Получен сигнал {signum!r}, завершаю работу...")
    # здесь можно добавить любую логику очистки, если нужно
    sys.exit(0)

# Register signal handlers
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP):
    signal.signal(sig, _shutdown)

if __name__ == "__main__":
    # выключаем перезагрузчик, чтобы ловить сигналы в одном процессе
    app.run_server(debug=False, use_reloader=False, port=8052)
