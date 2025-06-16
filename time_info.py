import json
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import base64
import signal
import sys
import plotly.graph_objects as go
from dash import callback_context
import io

# Initialize the Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# Create empty figures for initial layout
empty_fig = go.Figure()

# Create the layout
app.layout = html.Div([
    html.H1('Telegram Chat Statistics', style={'textAlign': 'center'}),
    
    # File upload component
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a JSON File')
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
    
    # Chat information
    html.Div(id='chat-info', style={'margin': '20px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
    
    # Graphs
    dcc.Graph(id='overall-graph', figure=empty_fig),
    dcc.Graph(id='daily-graph', figure=empty_fig),
    dcc.Graph(id='weekly-graph', figure=empty_fig),
    
    # Store for graph data
    dcc.Store(id='graph-data'),
    
    # Store for detailed stats visibility
    dcc.Store(id='stats-visible', data=False)
])

def create_figure_with_total(df, x_col, y_col, color_col, title, x_label, y_label, xaxis_config=None):
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add individual lines
    for sender in df[color_col].unique():
        sender_data = df[df[color_col] == sender]
        if sender_data[y_col].sum() > 0:  # Only add lines for senders with messages
            fig.add_trace(go.Scatter(
                x=sender_data[x_col],
                y=sender_data[y_col],
                name=sender,
                mode='lines'
            ))
    
    # Add total line
    total_data = df.groupby(x_col)[y_col].sum().reset_index()
    fig.add_trace(go.Scatter(
        x=total_data[x_col],
        y=total_data[y_col],
        name='Total',
        mode='lines',
        line=dict(width=3, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=True
    )
    
    if xaxis_config:
        fig.update_layout(xaxis=xaxis_config)
    
    return fig

def create_chat_info(messages, chat_data):
    total_messages = len(messages)
    sender_stats = messages.groupby('from').size()
    sender_percentages = (sender_stats / total_messages * 100).round(1)
    
    # Create basic info
    chat_type = "Private Chat" if chat_data['type'] == 'personal_chat' else "Group Chat"
    chat_name = chat_data['name']
    
    # Create detailed stats and sort by message count
    detailed_stats = []
    for sender, count in sender_stats.sort_values(ascending=False).items():
        percentage = sender_percentages[sender]
        detailed_stats.append(f"{sender}: {count} messages ({percentage}%)")
    
    # Create the info block
    info_blocks = [
        html.H3(f"{chat_type}: {chat_name}"),
        html.P(f"Total messages: {total_messages}")
    ]
    
    # Add detailed stats if it's a group chat or has few participants
    if chat_data['type'] == 'personal_chat' or len(sender_stats) <= 3:
        info_blocks.append(html.H4("Message Statistics:"))
        for stat in detailed_stats:
            info_blocks.append(html.P(stat))
    else:
        # Create collapsible section for group chats with many participants
        info_blocks.extend([
            html.Button('Show/Hide Detailed Statistics', id='toggle-stats', style={'margin': '10px'}),
            html.Div(
                [html.P(stat) for stat in detailed_stats],
                id='detailed-stats',
                style={'display': 'none'}
            )
        ])
    
    return html.Div(info_blocks)

def parse_contents(contents):
    if contents is None:
        return empty_fig, empty_fig, empty_fig, None, None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        data = json.loads(decoded.decode('utf-8'))
        messages = pd.DataFrame(data['messages'])
        
        # Convert date strings to datetime
        messages['date'] = pd.to_datetime(messages['date'])
        
        # Create time-based features
        messages['hour'] = messages['date'].dt.hour
        messages['minute'] = messages['date'].dt.minute
        messages['day_of_week'] = messages['date'].dt.day_name()
        messages['date_only'] = messages['date'].dt.date
        
        # Create 5-minute intervals
        messages['time_interval'] = messages['date'].dt.floor('5min').dt.time
        
        # Create time distribution data for daily pattern
        # First create a complete time range for all senders
        time_range = pd.date_range(
            start=pd.Timestamp('2000-01-01 00:00:00'),
            end=pd.Timestamp('2000-01-01 23:55:00'),
            freq='5min'
        ).time
        
        # Create a DataFrame with all time intervals and senders
        complete_times = pd.DataFrame([
            {'time_interval': time, 'from': sender}
            for time in time_range
            for sender in messages['from'].unique()
        ])
        
        # Get actual message counts
        time_dist = messages.groupby(['time_interval', 'from']).size().reset_index(name='count')
        
        # Merge with complete time range
        time_dist = complete_times.merge(
            time_dist,
            on=['time_interval', 'from'],
            how='left'
        ).fillna(0)
        
        # Sort by time and sender
        time_dist['time_interval_dt'] = pd.to_datetime(time_dist['time_interval'].astype(str))
        time_dist = time_dist.sort_values(['time_interval_dt', 'from'])
        time_dist = time_dist.drop('time_interval_dt', axis=1)
        
        # Create time distribution data for overall pattern (daily)
        messages['date_interval'] = messages['date'].dt.floor('D')
        date_dist = messages.groupby(['date_interval', 'from']).size().reset_index(name='count')
        
        # Create a complete date range from min to max date
        date_range = pd.date_range(
            start=messages['date'].min().floor('D'),
            end=messages['date'].max().floor('D'),
            freq='D'
        )
        
        # Get only senders who have at least one message
        active_senders = messages.groupby('from').size().reset_index(name='count')
        active_senders = active_senders[active_senders['count'] > 0]['from'].tolist()
        
        # Create a DataFrame with all dates and active senders
        complete_dates = pd.DataFrame([
            {'date_interval': date, 'from': sender}
            for date in date_range
            for sender in active_senders
        ])
        
        # Merge with actual data, filling missing values with 0
        date_dist = complete_dates.merge(
            date_dist,
            on=['date_interval', 'from'],
            how='left'
        ).fillna(0)
        
        # Remove rows where all senders have 0 messages
        date_dist = date_dist[date_dist.groupby('date_interval')['count'].transform('sum') > 0]
        
        date_dist = date_dist.sort_values(['date_interval', 'from'])
        
        # Create figures
        overall_fig = create_figure_with_total(
            date_dist, 'date_interval', 'count', 'from',
            'Number of Messages Over Time by Sender',
            'Date', 'Number of Messages',
            {'tickangle': 45}
        )
        
        daily_fig = create_figure_with_total(
            time_dist, 'time_interval', 'count', 'from',
            'Number of Messages by Time of Day (5-minute intervals) by Sender',
            'Time of Day', 'Number of Messages',
            {
                'tickformat': '%H:%M',
                'tickangle': 45,
                'tickmode': 'array',
                'ticktext': [f'{hour:02d}:00' for hour in range(24)],
                'tickvals': [pd.Timestamp(f'2000-01-01 {hour:02d}:00:00').time() for hour in range(24)]
            }
        )
        
        weekly_fig = px.histogram(
            messages,
            x='day_of_week',
            color='from',
            title='Number of Messages by Day of Week by Sender',
            labels={'day_of_week': 'Day of Week', 'count': 'Number of Messages', 'from': 'Sender'},
            category_orders={'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
        )
        
        graph_data = {
            'overall': overall_fig.to_json(),
            'daily': daily_fig.to_json(),
            'weekly': weekly_fig.to_json()
        }
        
        # Create chat info
        chat_info = create_chat_info(messages, data)
        
        return overall_fig, daily_fig, weekly_fig, graph_data, chat_info
        
    except Exception as e:
        return empty_fig, empty_fig, empty_fig, None, None

@app.callback(
    [Output('overall-graph', 'figure'),
     Output('daily-graph', 'figure'),
     Output('weekly-graph', 'figure'),
     Output('graph-data', 'data'),
     Output('chat-info', 'children')],
    [Input('upload-data', 'contents')],
    [State('graph-data', 'data')]
)
def update_graphs(contents, graph_data):
    if contents is None:
        return empty_fig, empty_fig, empty_fig, None, None
    
    return parse_contents(contents)

@app.callback(
    Output('detailed-stats', 'style'),
    Input('toggle-stats', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_stats(n_clicks):
    if n_clicks is None:
        return {'display': 'none'}
    return {'display': 'block' if n_clicks % 2 == 1 else 'none'}

def _shutdown(signum, frame):
    print(f"Получен сигнал {signum!r}, завершаю работу...")
    # здесь можно добавить любую логику очистки, если нужно
    sys.exit(0)

# Register signal handlers
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP):
    signal.signal(sig, _shutdown)

if __name__ == "__main__":
    # выключаем перезагрузчик, чтобы ловить сигналы в одном процессе
    app.run(debug=True, use_reloader=True, port=8051)