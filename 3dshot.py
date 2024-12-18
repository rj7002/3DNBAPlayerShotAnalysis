from datetime import date
from datetime import datetime
import requests
import streamlit as st
import plotly.express as px
import pandas as pd
from courtCoordinates import CourtCoordinates
# from basketballShot import BasketballShot
# from courtCoordinates2 import CourtCoordinates2
# from basketballShot2 import BasketballShot2
import pandas as pd
# from sportsdataverse.nba.nba_pbp import espn_nba_pbp
# from sportsdataverse.wnba.wnba_pbp import espn_wnba_pbp
import plotly.graph_objects as go  # Import Plotly graph objects separately
# import time
# import re
# import sportsdataverse
# from streamlit_plotly_events import plotly_events
from datetime import datetime, timedelta
from random import randint

from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.endpoints import ShotChartDetail
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import commonplayerinfo
import numpy as np
from streamlit_plotly_events import plotly_events

# --- Streamlit Page Config ---
st.set_page_config(layout='wide', page_title="NBA Shot Analysis", page_icon="🏀")

# --- Page Header ---
st.markdown(
    """
    <h1 style="text-align: center; font-size: 72px; color: red;">🏀 NBA Shot Analysis 🏀</h1>
    <p style="text-align: center; font-size: 18px; color: #555;">
        Analyze NBA players' shooting performances and shot tendencies. Data available from 1996 to present.
    </p>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    <h1 style="text-align: center; font-size: 25px; color: red;">🏀 NBA Shot Analysis 🏀</h1>
    """,
    unsafe_allow_html=True
)
year = datetime.now().year
month = datetime.now().month
currentyear = year
if month >= 10 and month <=12:  # NBA season starts in October
    currentyear=currentyear+1
else:
        currentyear=currentyear

def display_player_image(player_id, width2, caption2):
    # Construct the URL for the player image using the player ID
    image_url = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"
    
    # Check if the image URL returns a successful response
    response = requests.head(image_url)
    
    if response.status_code == 200:
        # If image is available, display it
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{caption2}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    
        # st.image(image_url, width=width2, caption=caption2)
    else:
        image_url = "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{"Image Unavailable"}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
def get_player_season_range(player_id):
    player_stats2 = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    player_stats = player_stats2.get_data_frames()[0]
    if not player_stats.empty:
        player_stats = player_stats[player_stats["PERSON_ID"] == player_id]
        if not player_stats.empty:
            first_season = player_stats['FROM_YEAR'].tolist()[0] if player_stats['FROM_YEAR'].tolist() else '1996'
            last_season = player_stats['TO_YEAR'].tolist()[0] if player_stats['TO_YEAR'].tolist() else str(currentyear)
        else:
            first_season = '1996'
            last_season = str(currentyear)
    else:
        first_season = '1996'
        last_season = str(currentyear)
    return first_season, last_season

players2 = players.get_players()
currentyear = datetime.now().year
allplayers = []
for player in players2:
    allplayers.append(player['full_name'] + ' - ' + str(player['id']))

selected_player = st.selectbox('Select a player',allplayers)
parts = selected_player.split(' - ')
id = int(parts[-1])

first_season, last_season = get_player_season_range(id)
first_season = int(first_season)
last_season = int(last_season)
# st.write(first_season)
# st.write(last_season)
selected_seasons = st.multiselect('Select a season', range(first_season,last_season+1))
if selected_seasons:
    realseason = selected_seasons[0]
    realseasonandone = realseason+1
    fullrealseason = f'{str(realseason)}-{str(realseasonandone)[2:]}'
    court = CourtCoordinates(fullrealseason)
    court_lines_df = court.get_coordinates()
    # st.write(court_lines_df)
    fig = px.line_3d(
        data_frame=court_lines_df,
        x='x',
        y='y',
        z='z',
        line_group='line_group_id',
        color='line_group_id',
        color_discrete_map={
            'court': 'black',
            'hoop': '#e47041',
            'net': '#D3D3D3',
            'backboard': 'gray',
            'backboard2': 'gray',
            'free_throw_line': 'black',
            'hoop2':'#D3D3D3',
            'free_throw_line2': 'black',
            'free_throw_line3': 'black',
            'free_throw_line4': 'black',
            'free_throw_line5': 'black',
        }
    )
    fig.update_traces(hovertemplate=None, hoverinfo='skip', showlegend=False)
    fig.update_traces(line=dict(width=6))
    court_perimeter_bounds = np.array([[-250, 0, -0.2], [250, 0, -0.2], [250, 450, -0.2], [-250, 450, -0.2], [-250, 0, -0.2]])

    # Extract x, y, and z values for the mesh
    court_x = court_perimeter_bounds[:, 0]
    court_y = court_perimeter_bounds[:, 1]
    court_z = court_perimeter_bounds[:, 2]
    
    # Add a square mesh to represent the court floor at z=0
    fig.add_trace(go.Mesh3d(
        x=court_x,
        y=court_y,
        z=court_z,
        color='#d2a679',
        # opacity=0.5,
        name='Court Floor',
        hoverinfo='none',
        showscale=False
    ))
    fig.update_layout(    
        margin=dict(l=20, r=20, t=20, b=20),
        scene_aspectmode="data",
        height=600,
        scene_camera=dict(
            eye=dict(x=1.3, y=0, z=0.7)
        ),
        scene=dict(
            xaxis=dict(title='', showticklabels=False, showgrid=False),
            yaxis=dict(title='', showticklabels=False, showgrid=False),
            zaxis=dict(title='',  showticklabels=False, showgrid=False, showbackground=False, backgroundcolor='#d2a679'),
        ),
        showlegend=False,
        legend=dict(
            yanchor='top',
            y=0.05,
            x=0.2,
            xanchor='left',
            orientation='h',
            font=dict(size=15, color='gray'),
            bgcolor='rgba(0, 0, 0, 0)',
            title='',
            itemsizing='constant'
        )
    )
    # df2 = shotchart(selected_player,realseason)
    # Section for Context Measure
    col1, col2,col3 = st.columns(3)
    with col1:
        anim = st.checkbox('Animated',help='View animated shots')
        context_measures = [
            "PTS", "FGA", "FG3M", "FG3A", 
             "PTS_FB", "PTS_OFF_TOV", "PTS_2ND_CHANCE"
        ]
        context_measure = st.sidebar.selectbox("Select Context Measure", context_measures, index=context_measures.index('FGA'))
    
        # Section for Last N Games
        last_n_games = st.sidebar.number_input("Last N Games", min_value=0, max_value=82, value=10, step=1)
    
        # Section for Month
        # month = st.selectbox("Select Month", list(range(1, 13)), index=None)
    
        # Section for Period
        # period = st.selectbox("Select Period", list(range(1, 5)), index=None)
    
        # Section for Season Type
        season_types = ["Regular Season", "Pre Season", "Playoffs", "All Star"]
        season_type = st.sidebar.selectbox("Select Season Type", season_types, index=None)
    
        # Section for Team ID
        # Section for Vs Division (optional)
        vs_division_options = [
            "Atlantic", "Central", "Northwest", "Pacific", "Southeast", "Southwest", "East", "West"
        ]
        vs_division = st.sidebar.selectbox("Vs Division (optional)", vs_division_options, index=None, key="vs_division")
    
        # Section for Vs Conference (optional)
    with col2:
        vids = st.checkbox('Highlights',help='Click on a shot to view the highlight (Highlights only available starting in 2014-15)')
        vs_conference_options = ["East", "West"]
        vs_conference = st.sidebar.selectbox("Vs Conference (optional)", vs_conference_options, index=None, key="vs_conference")
    
        # Section for Start Period (optional)
        # start_period = st.number_input("Start Period (optional)", min_value=1, value=None, step=1)
    
        # Section for Season Segment (Pre/Post All-Star)
        season_segment_options = ["Post All-Star", "Pre All-Star"]
        season_segment = st.sidebar.selectbox("Season Segment (optional)", season_segment_options, index=None)
    
        # Section for Point Differential (optional)
        point_diff = st.sidebar.number_input("Point Differential (optional)", min_value=-50, max_value=50, value=None, step=1)
    
        # Section for Outcome (Win or Loss)
        outcome = st.sidebar.selectbox("Outcome (optional)", ["W", "L"], index=None)
    
        # Section for Location (Home or Road)
        location = st.sidebar.selectbox("Location (optional)", ["Home", "Road"], index=None)
    
    
    
    with col3:
        fgperc = st.checkbox('FG%',help='View hot zones')
        # Section for Game Segment (First Half, Overtime, Second Half)
    
        # Section for Game ID (optional)
        # game_id = st.text_input("Game ID (optional)", value=None, max_chars=10)
    
        game_segment = st.sidebar.selectbox("Game Segment (optional)", ["First Half", "Overtime", "Second Half"], index=None)
    
    
        # Section for End Period (optional)
        # end_period = st.number_input("End Period (optional)", min_value=1, value=None, step=1)
    
        # Section for Date From (optional)
        date_from = st.sidebar.date_input("Date From (optional)", None)
    
        # Section for Date To (optional)
        date_to = st.sidebar.date_input("Date To (optional)", None)
    
        # Section for Context Filter (optional)
        # context_filter = st.text_input("Context Filter (optional)", value=None)
    
        # Section for Clutch Time (Last 5, 4, 3, etc.)
        clutch_time_options = [
            "Last 5 Minutes", "Last 4 Minutes", "Last 3 Minutes", "Last 2 Minutes",
            "Last 1 Minute", "Last 30 Seconds", "Last 10 Seconds"
        ]
        clutch_time = st.sidebar.selectbox("Clutch Time (optional)", clutch_time_options, index=None)
    
        # Section for Ahead/Behind (optional)
        ahead_behind_options = [
            "Ahead or Behind", "Ahead or Tied", "Behind or Tied"
        ]
        ahead_behind = st.sidebar.selectbox("Ahead/Behind (optional)", ahead_behind_options, index=None)
    df = pd.DataFrame()
    for selected_season in selected_seasons:
        nextseason = selected_season+1
        nextstr = str(nextseason)[2:]
        realseason = str(selected_season) + "-" + nextstr
        params = {
            "player_id": id,
            "season_nullable": realseason,
            "team_id": 0,
            "context_measure_simple": context_measure,
            "last_n_games": last_n_games,
            "season_type_all_star": season_type,
            "vs_division_nullable": vs_division,
            "vs_conference_nullable": vs_conference,
            "season_segment_nullable": season_segment,
            "point_diff_nullable": point_diff,
            "outcome_nullable": outcome,
            "location_nullable": location,
            "game_segment_nullable": game_segment,
            "date_from_nullable": date_from,
            "date_to_nullable": date_to,
            "clutch_time_nullable": clutch_time,
            "ahead_behind_nullable": ahead_behind
        }
    
        # Remove all parameters that are None
        params = {key: value for key, value in params.items() if value is not None}
        
        # Create ShotChartDetail instance with filtered parameters
        # if st.button('Submit'):
        shotchartdata = shotchartdetail.ShotChartDetail(**params)
        all_shot_data = shotchartdata.get_data_frames()[0]
        df = pd.concat([df, all_shot_data], ignore_index=True)
    # st.write(df.columns)
    unique_periods = df['PERIOD'].unique()
    Quarter = st.sidebar.toggle('Quarter')
    if Quarter == 1:
        quart = st.sidebar.multiselect('',unique_periods)
    ShotDist = st.sidebar.toggle('Shot Distance')
    if ShotDist == 1:
        shotdistbool = True
        # shotdistance = st.sidebar.slider("Shot Distance", 0, 40)
        shotdistance_min, shotdistance_max = st.sidebar.slider("Shot Distance", 0, 92, (0, 92))
    ShotType = st.sidebar.toggle('Shot Type')
    if ShotType == 1:
        shottypebool = True
        shottypes = df['ACTION_TYPE'].unique()
        shottype = st.sidebar.multiselect('', shottypes)
        # if shottype == 'Jump Shot':
        #     jumpshottype = st.sidebar.multiselect('', ['Stepback Jump shot', 'Running Pull-Up Jump Shot','Turnaround Fadeaway shot','Fadeaway Jump Shot','Pullup Jump shot','Jump Bank Shot','Jump Shot'])
        #     finaltype = jumpshottype
        # elif shottype == 'Layup':
        #     layuptype = st.sidebar.multiselect('', ['Layup Shot', 'Running Finger Roll Layup Shot','Cutting Layup Shot','Driving Layup Shot','Running Layup Shot','Alley Oop Layup shot','Tip Layup Shot','Reverse Layup Shot','Driving Reverse Layup Shot','Running Reverse Layup Shot'])
        #     finaltype = layuptype
        # elif shottype == 'Dunk':
        #     dunktype = st.sidebar.multiselect('', ['Running Dunk Shot', 'Cutting Dunk Shot','Running Reverse Dunk Shot','Running Alley Oop Dunk Shot','Dunk Shot','Tip Dunk Shot'])    
        #     finaltype = dunktype
        # elif shottype == 'Other':
        #     othertype = st.sidebar.multiselect('', ['Driving Floating Jump Shot', 'Floating Jump shot','Driving Floating Bank Jump Shot','Driving Bank Hook Shot','Driving Hook Shot','Turnaround Hook Shot','Hook Shot'])
        #     finaltype = othertype
    Teams = st.sidebar.toggle('Teams')
    if Teams == 1:
        teamtype = st.sidebar.multiselect('', ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'])
    CourtLoc = st.sidebar.toggle('Court Location')
    if CourtLoc == 1:
        courtloctypes = df['SHOT_ZONE_AREA'].unique()
        courtloc = st.sidebar.multiselect('',courtloctypes)
    Time = st.sidebar.toggle('Time')
    if Time == 1:
        timemin, timemax = st.sidebar.slider("Time Remaining (Minutes)", 0, 15, (0, 15))
    if ShotType:  # Check if ShotType checkbox is selected
        df = df[df['ACTION_TYPE'].isin(shottype)]
        # Plot makes in green
    if ShotDist == 1:
        df = df[(df['SHOT_DISTANCE'] >= shotdistance_min) & (df['SHOT_DISTANCE'] <= shotdistance_max)]
    if Teams:
        df = df[(df['VTM'].isin(teamtype)) | (df['HTM'].isin(teamtype))]
    if CourtLoc:
        df = df[df['SHOT_ZONE_AREA'].isin(courtloc)]
    if Time:
        df = df[(df['MINUTES_REMAINING'] >= timemin) & (df['MINUTES_REMAINING'] <= timemax)]
    if Quarter:
        df = df[df['PERIOD'].isin(quart)]
    if vids != 1 and anim != 1:
        Make = st.sidebar.checkbox('Make Shot Paths',value=True)
        Miss = st.sidebar.checkbox('Miss Shot Paths')


    if selected_seasons:
        if len(df) > 0:
            if len(df) > 500:
                st.warning('There is over 500 shots being plotted so the plot may be slow')
            st.success('Data Found')
            # df = df.head(500)
            if vids != 1 and anim != 1:
               
                dfmiss = df[df['SHOT_MADE_FLAG'] == 0]
                dfmake = df[df['SHOT_MADE_FLAG'] == 1]
                if Make:
                    x_values = []
                    y_values = []
                    z_values = []
                    for index, row in dfmake.iterrows():
                        
                        
                    
                        x_values.append(-row['LOC_X'])
                        # Append the value from column 'x' to the list
                        y_values.append(row['LOC_Y']+45)
                        z_values.append(0)
                
                
                
                    x_values2 = []
                    y_values2 = []
                    z_values2 = []
                    for index, row in dfmake.iterrows():
                        # Append the value from column 'x' to the list
                    
                
                        x_values2.append(court.hoop_loc_x)
                
                        y_values2.append(court.hoop_loc_y)
                        z_values2.append(100)
                
                    import numpy as np
                    import plotly.graph_objects as go
                    import streamlit as st
                    import math
                    def calculate_distance(x1, y1, x2, y2):
                        """Calculate the distance between two points (x1, y1) and (x2, y2)."""
                        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                    def generate_arc_points(p1, p2, apex, num_points=100):
                        """Generate points on a quadratic Bezier curve (arc) between p1 and p2 with an apex."""
                        t = np.linspace(0, 1, num_points)
                        x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
                        y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
                        z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
                        return x, y, z
                
                    # Example lists of x and y coordinates
                    x_coords = x_values
                    y_coords = y_values
                    z_value = 0  # Fixed z value
                    x_coords2 = x_values2
                    y_coords2 = y_values2
                    z_value2 = 100
                if Miss:
                    mx_values = []
                    my_values = []
                    mz_values = []
                    for index, row in dfmiss.iterrows():
                        
                        
                    
                        mx_values.append(-row['LOC_X'])
                        # Append the value from column 'x' to the list
                        my_values.append(row['LOC_Y']+45)
                        mz_values.append(0)
                
                
                
                    mx_values2 = []
                    my_values2 = []
                    mz_values2 = []
                    for index, row in dfmiss.iterrows():
                        # Append the value from column 'x' to the list
                    
                
                        mx_values2.append(court.hoop_loc_x)
                
                        my_values2.append(court.hoop_loc_y)
                        mz_values2.append(100)
                
                    import numpy as np
                    import plotly.graph_objects as go
                    import streamlit as st
                    import math
                    def calculate_distance(x1, y1, x2, y2):
                        """Calculate the distance between two points (x1, y1) and (x2, y2)."""
                        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                    def generate_arc_points(p1, p2, apex, num_points=100):
                        """Generate points on a quadratic Bezier curve (arc) between p1 and p2 with an apex."""
                        t = np.linspace(0, 1, num_points)
                        x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
                        y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
                        z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
                        return x, y, z
                
                    # Example lists of x and y coordinates
                    mx_coords = mx_values
                    my_coords = my_values
                    mz_value = 0  # Fixed z value
                    mx_coords2 = mx_values2
                    my_coords2 = my_values2
                    mz_value2 = 100
            with col1:
                if anim == 1:
                    newdf = df.copy()
                    newdf = newdf[newdf['SHOT_MADE_FLAG'] == 1]
                    newdf = newdf[newdf['SHOT_DISTANCE'] > 3]
                    # if len(newdf) > 150: 
                    #     st.error(f'Too many shots. Only showing first 150 shots.')
                    #     newdf = newdf.head(150)
                    # else:
                    #     newdf = newdf
                    if len(newdf) >= 100:
                            default = 10
                    elif len(newdf) >= 75:
                        default = 8
                    elif len(newdf) >= 50:
                        default = 5
                    elif len(newdf) >= 20:
                        default = 3
                    else:
                        default = 1
                    shotgroup = st.number_input("Number of Shots Together", min_value=1, max_value=10, step=1, value=default)
                    court_perimeter_bounds = np.array([[-250, 0, 0], [250, 0, 0], [250, 450, 0], [-250, 450, 0], [-250, 0, 0]])
                    
                    # Extract x, y, and z values for the mesh
                    court_x = court_perimeter_bounds[:, 0]
                    court_y = court_perimeter_bounds[:, 1]
                    court_z = court_perimeter_bounds[:, 2]
                    
                    # Add a square mesh to represent the court floor at z=0
                    fig.add_trace(go.Mesh3d(
                        x=court_x,
                        y=court_y,
                        z=court_z-1,
                        color='#d2a679',
                        opacity=1,
                        name='Court Floor',
                        hoverinfo='none',
                        showscale=False
                    ))
                    # hover_data = newdf.apply(lambda row: f"""
                    #     <b>Player:</b> {row['fullName']}<br>
                    #     <b>Game Date:</b> {row['gameDate']}<br>
                    #     <b>Game:</b> {row['TeamName']} vs {row['OpponentName']}<br>
                    #     <b>Half:</b> {row['period'][-1]}<br>
                    #     <b>Time:</b> {row['clock']}<br>
                    #     <b>Result:</b> {'Made' if row['success'] else 'Missed'}<br>
                    #     <b>Shot Distance:</b> {row['shotDist']} ft<br>
                    #     <b>Shot Type:</b> {row['actionType']} ({row['subType']})<br>
                    #     <b>Shot Clock:</b> {row['shotClock']}<br>
                    #     <b>Assisted by:</b> {row['assisterName']}<br>
                    # """, axis=1)
                   
                    court_perimeter_lines = court_lines_df[court_lines_df['line_id'] == 'outside_perimeter']
                    three_point_lines = court_lines_df[court_lines_df['line_id'] == 'three_point_line']
                    backboard = court_lines_df[court_lines_df['line_id'] == 'backboard']
                    backboard2 = court_lines_df[court_lines_df['line_id'] == 'backboard2']
                    freethrow = court_lines_df[court_lines_df['line_id'] == 'free_throw_line']
                    freethrow2 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line2']
                    freethrow3 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line3']
                    freethrow4 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line4']
                    freethrow5 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line5']
                    hoop = court_lines_df[court_lines_df['line_id'] == 'hoop']
                    hoop2 = court_lines_df[court_lines_df['line_id'] == 'hoop2']
                    
                    
                    
                    
                    
                    
                    
                    # Add court lines to the plot (3D scatter)
                    fig.add_trace(go.Scatter3d(
                        x=court_perimeter_lines['x'],
                        y=court_perimeter_lines['y'],
                        z=np.zeros(len(court_perimeter_lines)),  # Place court lines on the floor
                        mode='lines',
                        line=dict(color='black', width=6),
                        name="Court Perimeter",
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=hoop['x'],
                        y=hoop['y'],
                        z=hoop['z'],  # Place 3-point line on the floor
                        mode='lines',
                        line=dict(color='#e47041', width=6),
                        name="Hoop",
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=hoop2['x'],
                        y=hoop2['y'],
                        z=hoop2['z'],  # Place 3-point line on the floor
                        mode='lines',
                        line=dict(color='#D3D3D3', width=6),
                        name="Backboard",
                        hoverinfo='none'
                    ))
                    # Add the 3-point line to the plot
                    fig.add_trace(go.Scatter3d(
                        x=backboard['x'],
                        y=backboard['y'],
                        z=backboard['z'],  # Place 3-point line on the floor
                        mode='lines',
                        line=dict(color='grey', width=6),
                        name="Backboard",
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=backboard2['x'],
                        y=backboard2['y'],
                        z=backboard2['z'],  # Place 3-point line on the floor
                        mode='lines',
                        line=dict(color='grey', width=6),
                        name="Backboard",
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=three_point_lines['x'],
                        y=three_point_lines['y'],
                        z=np.zeros(len(three_point_lines)),  # Place 3-point line on the floor
                        mode='lines',
                        line=dict(color='black', width=6),
                        name="3-Point Line",
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=freethrow['x'],
                        y=freethrow['y'],
                        z=np.zeros(len(freethrow)),  # Place court lines on the floor
                        mode='lines',
                        line=dict(color='black', width=6),
                        name="Court Perimeter",
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=freethrow2['x'],
                        y=freethrow2['y'],
                        z=np.zeros(len(freethrow2)),  # Place court lines on the floor
                        mode='lines',
                        line=dict(color='black', width=6),
                        name="Court Perimeter",
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=freethrow3['x'],
                        y=freethrow3['y'],
                        z=np.zeros(len(freethrow3)),  # Place court lines on the floor
                        mode='lines',
                        line=dict(color='black', width=6),
                        name="Court Perimeter",
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=freethrow4['x'],
                        y=freethrow4['y'],
                        z=np.zeros(len(freethrow4)),  # Place court lines on the floor
                        mode='lines',
                        line=dict(color='black', width=6),
                        name="Court Perimeter",
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=freethrow5['x'],
                        y=freethrow5['y'],
                        z=np.zeros(len(freethrow5)),  # Place court lines on the floor
                        mode='lines',
                        line=dict(color='black', width=6),
                        name="Court Perimeter",
                        hoverinfo='none'
                    ))
                    x_values = []
                    y_values = []
                    z_values = []
                    # dfmiss = df[df['SHOT_MADE_FLAG'] == 0]
                    # df = df[df['SHOT_MADE_FLAG'] == 1]
                    
            
                    for index, row in newdf.iterrows():
                        
                        
                    
                        x_values.append(-row['LOC_X'])
                        # Append the value from column 'x' to the list
                        y_values.append(row['LOC_Y']+45)
                        z_values.append(0)
                    
                    
                    
                    x_values2 = []
                    y_values2 = []
                    z_values2 = []
                    import math
                    for index, row in newdf.iterrows():
                        # Append the value from column 'x' to the list
                    
                    
                        x_values2.append(court.hoop_loc_x)
                    
                        y_values2.append(court.hoop_loc_y)
                        z_values2.append(100)
                    
                    def calculate_distance(x1, y1, x2, y2):
                        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Function to generate arc points
                    def generate_arc_points(p1, p2, apex, num_points=100):
                        t = np.linspace(0, 1, num_points)
                        x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
                        y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
                        z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
                        return x, y, z
                    
                    
                   
                    
                    frames = []
                    num_points = 100  # Increase this for more resolution
                    segment_size = 20
                    # Function to process shots in batches
                    def process_shots_in_batches(shotdf, batch_size=3):
                        for batch_start in range(0, len(shotdf), batch_size):
                            batch_end = min(batch_start + batch_size, len(shotdf))
                            yield shotdf[batch_start:batch_end]
                    
                    # Generate frames for each batch
                    for batch in process_shots_in_batches(newdf, batch_size=10):
                        for t in np.linspace(0, 1, 8):  # Adjust for smoothness
                            frame_data = []
                            
                            for _, row in batch.iterrows():
                                x1, y1 = int(row['LOC_X']), int(row['LOC_Y'])
                                x2, y2 = court.hoop_loc_x, court.hoop_loc_y
                                p2 = np.array([x1, y1, 0])
                                p1 = np.array([x2, y2, 100])
                    
                                # Arc height based on shot distance
                                h = (150 if row['SHOT_DISTANCE'] <= 15 else
                                     200 if row['SHOT_DISTANCE'] <= 25 else
                                     250 if row['SHOT_DISTANCE'] <= 30 else
                                     300 if row['SHOT_DISTANCE'] <= 50 else
                                     325)
                                apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])
                                x, y, z = generate_arc_points(p2, p1, apex, num_points)
                    
                                # Calculate the start and end of the moving segment
                                total_points = len(x)
                                start_index = int(t * (total_points - segment_size))
                                end_index = start_index + segment_size
                    
                                # Ensure indices are within bounds
                                start_index = max(0, start_index)
                                end_index = min(total_points, end_index)
                    
                                segment_x = x[start_index:end_index]
                                segment_y = y[start_index:end_index]
                                segment_z = z[start_index:end_index]
                    
                                frame_data.append(go.Scatter3d(
                                    x=segment_x, y=segment_y, z=segment_z,
                                    mode='lines', line=dict(width=6, color='green'),
                                    hoverinfo='text', hovertext=row.get('hover_text', '')
                                ))
                              
                            
                            frames.append(go.Frame(data=frame_data))
                    
                    
                    # Add an initial empty trace for layout
                    fig.add_trace(go.Scatter3d(x=[], y=[], z=[]))
                    
                    # Empty frame at the end for clearing the court
                    fig.add_trace(go.Scatter3d(x=[], y=[], z=[]))
                    empty_frame_data = []
                    for i in range(0,10):
                        empty_frame_data.append(go.Scatter3d(
                        x=[0], y=[0], z=[0],
                        mode='lines', line=dict(width=6, color='rgba(255, 0, 0, 0)')
                    ))
                    
                    frames.append(go.Frame(data=empty_frame_data))
                                        
                    
                    # Add frames to the figure
                    fig.frames = frames
                    
                    
                    
                    # Layout with animation controls
                    fig.update_layout(
                        updatemenus=[
                            dict(type="buttons",
                                 showactive=False,
                                 buttons=[
                                     dict(label="Play",
                                          method="animate",
                                          args=[None, {"frame": {"duration": 0.000000000001, "redraw": True}, "fromcurrent": True}]),
                                     dict(label="Pause",
                                          method="animate",
                                          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                                 ])
                        ],
                        # # scene=dict(
                        # #     # xaxis=dict(range=[-25, 25], title="X"),
                        # #     # yaxis=dict(range=[-50, 50], title="Y"),
                        #     zaxis=dict(range=[0, 175], title="Z"),
                        # #     # aspectratio=dict(x=1, y=1, z=0.5),
                        # # ),
                    )
                elif vids == 1:
                    court = CourtCoordinates(fullrealseason)
                    court_lines_df = court.get_coordinates()
                    fig = px.line_3d(
                        data_frame=court_lines_df,
                        x='x',
                        y='y',
                        z='z',
                        line_group='line_group_id',
                        color='line_group_id',
                        color_discrete_map={
                            'court': 'black',
                            'hoop': '#e47041',
                            'net': '#D3D3D3',
                            'backboard': 'gray',
                            'backboard2': 'gray',
                            'free_throw_line': 'black',
                            'hoop2':'#D3D3D3',
                            'free_throw_line2': 'black',
                            'free_throw_line3': 'black',
                            'free_throw_line4': 'black',
                            'free_throw_line5': 'black',
                        }
                    )
                    court_perimeter_bounds = np.array([[-250, 0, -0.2], [250, 0, -0.2], [250, 450, -0.2], [-250, 450, -0.2], [-250, 0, -0.2]])

                    # Extract x, y, and z values for the mesh
                    court_x = court_perimeter_bounds[:, 0]
                    court_y = court_perimeter_bounds[:, 1]
                    court_z = court_perimeter_bounds[:, 2]
                    
                    # Add a square mesh to represent the court floor at z=0
               
                    
                    # Update hovertemplate to show detailed information when hovering
                    # fig.update_traces(
                    #     hovertemplate="<b>Player:</b> %{customdata[0]}<br>" +  # Player name
                    #                   "<b>Team:</b> %{customdata[1]}<br>" +    # Team
                    #                   "<b>Assist Type:</b> %{customdata[2]}<br>" +  # Assist type
                    #                   "<b>Pass Distance:</b> %{customdata[3]} ft<br>",  # Pass distance
                    # )
                    fig.update_traces(line=dict(width=6))
                    fig.add_trace(go.Mesh3d(
                        x=court_x,
                        y=court_y,
                        z=court_z,
                        color='#d2a679',
                        # opacity=0.5,
                        name='Court Floor',
                        hoverinfo='none',
                        showscale=False
                    ))
                    fig.update_layout(    
                       margin=dict(l=0, r=0, t=0, b=0),
                        scene_aspectmode="data",
                        scene=dict(
                            xaxis=dict(title='', showticklabels=False, showgrid=False),
                            yaxis=dict(title='', showticklabels=False, showgrid=False),
                            zaxis=dict(title='',  showticklabels=False, showgrid=False, showbackground=False, backgroundcolor='#d2a679'),
                        ),
                        showlegend=False,
                    
                    )
                    fig.update_traces(hovertemplate=None, hoverinfo='skip', showlegend=False)
                    
                    # st.write(df)
                    for i, row in df.iterrows():
                            if row['SHOT_MADE_FLAG'] == 1:
                                    s = 'circle-open'
                                    s2 = 'circle'
                                    size = 9
                                    color = 'green'
                            else:
                                s = 'cross'
                                s2 = 'cross'
                                size = 10
                                color = 'red'
                            date_str = row['GAME_DATE']
                            game_date = datetime.strptime(date_str, "%Y%m%d")
                            formatted_date = game_date.strftime("%m/%d/%Y")
                            hovertemplate = f"Date: {formatted_date}<br>Game: {row['HTM']} vs {row['VTM']}<br>Result: {row['EVENT_TYPE']}<br>Shot Type: {row['ACTION_TYPE']}<br>Distance: {row['SHOT_DISTANCE']} ft {row['SHOT_TYPE']}<br>Quarter: {row['PERIOD']}<br>Time: {row['MINUTES_REMAINING']}:{row['SECONDS_REMAINING']}"

                    
                            fig.add_trace(go.Scatter3d(
                                x=[-row['LOC_X']],  # Single point, so wrap in a list
                                y=[row['LOC_Y']+45],  # Single point, so wrap in a list
                                z=[0],  # z is set to 0 for each point (flat 2D plot in the XY plane)
                                marker=dict(size=size, symbol=s2, color=color),  # Customize marker size, symbol, and color
                                name=f'Endpoint {i + 1}',  # Dynamically create a name for each trace
                                hoverinfo='text',
                                hovertemplate=hovertemplate
                            ))
                else:
                    if Make:
                        for i in range(len(dfmake)):
                            x1 = x_coords[i]
                            y1 = y_coords[i]
                            x2 = x_coords2[i]
                            y2 = y_coords2[i]
                            # Define the start and end points
                            p2 = np.array([x1, y1, z_value])
                            p1 = np.array([x2, y2, z_value2])
                            
                            # Apex will be above the line connecting p1 and p2
                            distance = calculate_distance(x1, y1, x2, y2)
            
                            if dfmake['SHOT_MADE_FLAG'].iloc[i] == 1:
                                s = 'circle-open'
                                s2 = 'circle'
                                size = 9
                                color = 'green'
                            else:
                                s = 'cross'
                                s2 = 'cross'
                                size = 10
                                color = 'red'
                            date_str = dfmake['GAME_DATE'].iloc[i]
                            game_date = datetime.strptime(date_str, "%Y%m%d")
                            formatted_date = game_date.strftime("%m/%d/%Y")
                            if int(df['SECONDS_REMAINING'].iloc[i]) < 10:
                                dfmake['SECONDS_REMAINING'].iloc[i] = '0' + str(df['SECONDS_REMAINING'].iloc[i])
                            hovertemplate= f"Date: {formatted_date}<br>Game: {dfmake['HTM'].iloc[i]} vs {dfmake['VTM'].iloc[i]}<br>Result: {dfmake['EVENT_TYPE'].iloc[i]}<br>Shot Type: {dfmake['ACTION_TYPE'].iloc[i]}<br>Distance: {dfmake['SHOT_DISTANCE'].iloc[i]} ft {dfmake['SHOT_TYPE'].iloc[i]}<br>Quarter: {dfmake['PERIOD'].iloc[i]}<br>Time: {dfmake['MINUTES_REMAINING'].iloc[i]}:{dfmake['SECONDS_REMAINING'].iloc[i]}"
            
                            if dfmake['SHOT_DISTANCE'].iloc[i] > 3:
                                if dfmake['SHOT_DISTANCE'].iloc[i] > 50:
                                    h = randint(255,305)
                                elif dfmake['SHOT_DISTANCE'].iloc[i] > 30:
                                    h = randint(230,280)
                                elif dfmake['SHOT_DISTANCE'].iloc[i] > 25:
                                    h = randint(180,230)
                                elif dfmake['SHOT_DISTANCE'].iloc[i] > 15:
                                    h = randint(180,230)
                                else:
                                    h = randint(130,160)
                            
                                apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])  # Adjust apex height as needed
                                
                                # Generate arc points
                                x, y, z = generate_arc_points(p1, p2, apex)
                                fig.add_trace(go.Scatter3d(
                                            x=x, y=y, z=z,
                                            mode='lines',
                                            line=dict(width=8,color = color),
                                            opacity =0.5,
                                            # name=f'Arc {i + 1}',
                                            # hoverinfo='text',
                                            hovertemplate=hovertemplate
                                        ))

                    if Miss:
                        for i in range(len(dfmiss)):
                            mx1 = mx_coords[i]
                            my1 = my_coords[i]
                            mx2 = mx_coords2[i]
                            my2 = my_coords2[i]
                            # Define the start and end points
                            mp2 = np.array([mx1, my1, mz_value])
                            mp1 = np.array([mx2, my2, mz_value2])
                            
                            # Apex will be above the line connecting p1 and p2
                            distance = calculate_distance(mx1, my1, mx2, my2)
            
                            if dfmiss['SHOT_MADE_FLAG'].iloc[i] == 1:
                                s = 'circle-open'
                                s2 = 'circle'
                                size = 9
                                color = 'green'
                            else:
                                s = 'cross'
                                s2 = 'cross'
                                size = 10
                                color = 'red'
                            date_str = dfmiss['GAME_DATE'].iloc[i]
                            game_date = datetime.strptime(date_str, "%Y%m%d")
                            formatted_date = game_date.strftime("%m/%d/%Y")
                            if int(dfmiss['SECONDS_REMAINING'].iloc[i]) < 10:
                                dfmiss['SECONDS_REMAINING'].iloc[i] = '0' + str(dfmiss['SECONDS_REMAINING'].iloc[i])
                            hovertemplate= f"Date: {formatted_date}<br>Game: {dfmiss['HTM'].iloc[i]} vs {dfmiss['VTM'].iloc[i]}<br>Result: {dfmiss['EVENT_TYPE'].iloc[i]}<br>Shot Type: {dfmiss['ACTION_TYPE'].iloc[i]}<br>Distance: {dfmiss['SHOT_DISTANCE'].iloc[i]} ft {dfmiss['SHOT_TYPE'].iloc[i]}<br>Quarter: {dfmiss['PERIOD'].iloc[i]}<br>Time: {dfmiss['MINUTES_REMAINING'].iloc[i]}:{dfmiss['SECONDS_REMAINING'].iloc[i]}"
            
                            if dfmiss['SHOT_DISTANCE'].iloc[i] > 3:
                                if dfmiss['SHOT_DISTANCE'].iloc[i] > 50:
                                    h = randint(255,305)
                                elif dfmiss['SHOT_DISTANCE'].iloc[i] > 30:
                                    h = randint(230,280)
                                elif dfmiss['SHOT_DISTANCE'].iloc[i] > 25:
                                    h = randint(180,230)
                                elif dfmiss['SHOT_DISTANCE'].iloc[i] > 15:
                                    h = randint(180,230)
                                else:
                                    h = randint(130,160)
                            
                                mapex = np.array([0.5 * (mx1 + mx2), 0.5 * (my1 + my2), h])  # Adjust apex height as needed
                                
                                # Generate arc points
                                mx, my, mz = generate_arc_points(mp1, mp2, mapex)
                                fig.add_trace(go.Scatter3d(
                                            x=mx, y=my, z=mz,
                                            mode='lines',
                                            line=dict(width=8,color = color),
                                            opacity =0.5,
                                            # name=f'Arc {i + 1}',
                                            # hoverinfo='text',
                                            hovertemplate=hovertemplate
                                        ))
                        # Add start and end points
        
                        # fig.add_trace(go.Scatter3d(
                        #     x=[p2[0], p2[0]],
                        #     y=[p2[1], p2[1]],
                        #     z=[p2[2], p2[2]],
                        #     mode='markers',
                        #     marker=dict(size=size, symbol=s,color=color),
                        #     # name=f'Endpoints {i + 1}',
                        #     # hoverinfo='text',
                        #     hovertemplate=hovertemplate
                        # ))
                        # fig.add_trace(go.Scatter3d(
                        #     x=[p2[0], p2[0]],
                        #     y=[p2[1], p2[1]],
                        #     z=[p2[2], p2[2]],
                        #     mode='markers',
                        #     marker=dict(size=5, symbol=s2,color = color),
                        #     # name=f'Endpoints {i + 1}',
                        #     # hoverinfo='text',
                        #     hovertemplate=hovertemplate
        
                        # ))
            if anim != 1:
                for i, row in df.iterrows():
                    if row['SHOT_MADE_FLAG'] == 1:
                        s = 'circle-open'
                        s2 = 'circle'
                        size = 9
                        color = 'green'
                    else:
                        s = 'cross'
                        s2 = 'cross'
                        size = 10
                        color = 'red'
                    hovertemplate= f"Game: {row['HTM']} vs {row['VTM']}<br>Result: {row['EVENT_TYPE']}<br>Shot Type: {row['ACTION_TYPE']}<br>Distance: {row['SHOT_DISTANCE']} ft {row['SHOT_TYPE']}<br>Quarter: {row['PERIOD']}<br>Time: {row['MINUTES_REMAINING']}:{row['SECONDS_REMAINING']}"
            
                    fig.add_trace(go.Scatter3d(
                        x=[-row['LOC_X']],  # Single point, so wrap in a list
                        y=[row['LOC_Y']+45],  # Single point, so wrap in a list
                        z=[0],  # z is set to 0 for each point (flat 2D plot in the XY plane)
                        marker=dict(size=size, symbol=s, color=color),  # Customize marker size, symbol, and color
                        name=f'Endpoint {i + 1}',  # Dynamically create a name for each trace
                        hoverinfo='text',
                        hovertemplate=hovertemplate
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=[-row['LOC_X']],  # Single point, so wrap in a list
                        y=[row['LOC_Y']+45],  # Single point, so wrap in a list
                        z=[0],  # z is set to 0 for each point (flat 2D plot in the XY plane)
                        marker=dict(size=size, symbol=s2, color=color),  # Customize marker size, symbol, and color
                        name=f'Endpoint {i + 1}',  # Dynamically create a name for each trace
                        hoverinfo='text',
                        hovertemplate=hovertemplate
                    ))
            playerparts = selected_player.split(' - ')
            player = playerparts[0]
            made = len(df[df['SHOT_MADE_FLAG'] == 1])
            total = len(df)
            if selected_season == currentyear:
                shottype = 'is shooting'
            else:
                shottype = 'shot'
            if CourtLoc:
                courtloc = f'from the {courtloc}'
            else:
                courtloc = 'from the field'
            sentence_parts = [f"{player} {shottype} {round((made/total)*100,2)}% {courtloc}"]
        
            # Add filters to sentence based on their selection
        
            if selected_seasons:
                selected_season2 = selected_season+1
                season2 = str(selected_season2)
                season2 = season2[2:]
                sentence_parts.append(f"in {selected_seasons}")
            if season_type:
                sentence_parts.append(f"in the {season_type.lower()}")
            if last_n_games:
                sentence_parts.append(f"in the last {last_n_games} games")
            # if month:
            #     month_name = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][month - 1]
            #     sentence_parts.append(f"in {month_name}")
            # if opponent_team_id:
            #     sentence_parts.append(f"against team {opponent_team_id}")
            # if period:
            #     if period == 1:
            #         periodtype = 'the 1st quarter'
            #     elif period == 2:
            #         periodtype = 'the 2nd quarter'
            #     elif period == 3:
            #         periodtype = 'the 3rd quarter'
            #     elif period == 4:
            #         periodtype = 'the 4th quarter'
            #     else:
            #         periodtype = 'overtime'
                # sentence_parts.append(f"in {periodtype}")
            if vs_division:
                sentence_parts.append(f"vs the {vs_division} division")
            if vs_conference:
                if vs_conference == 'East':
                    realconf = 'the Eastern'
                else:
                    realconf = 'the Western'
                sentence_parts.append(f"vs {realconf} conference")
            # if start_period:
            #     sentence_parts.append(f"starting from period {start_period}")
            if season_segment:
                sentence_parts.append(f"{season_segment.lower()} break")
            if point_diff:
                sentence_parts.append(f"with a point differential of {point_diff}")
            if outcome:
                if outcome == 'W':
                    realoutcome = "wins"
                else:
                    realoutcome = 'losses'
                sentence_parts.append(f"in {realoutcome}")
            if location:
                sentence_parts.append(f"at {location.lower()}")
            if game_segment:
                sentence_parts.append(f"during the {game_segment.lower()}")
            if clutch_time:
                sentence_parts.append(f"in the {clutch_time.lower()} of games")
            if ahead_behind:
                sentence_parts.append(f"when {ahead_behind.lower()}")
            if ShotDist:
                sentence_parts.append(f"from {shotdistance_min} to {shotdistance_max} feet")
            if ShotType:
                sentence_parts.append(f"on {shottype}s")
            if Teams:
                sentence_parts.append(f"vs {teamtype}")
            if Quarter:
                if len(quart) > 1:
                    qpart = 'quarters'
                else:
                    qpart = 'quarter'
                sentence_parts.append(f"in {qpart} {quart}")
            if Time:
                sentence_parts.append(f"between {timemin} and {timemax} minutes")
            # Combine sentence parts into a full sentence
            sentence = " ".join(sentence_parts) + "."
            display_player_image(id,400,'')
            st.subheader(sentence)
            # st.subheader(f'{player} Shot Chart in {realseason}')
            coli1, coli2 = st.columns(2)
            with coli1:
                if vids == 1:
                    import requests
                    
                    selected_points = plotly_events(fig, hover_event=False, click_event=True)
                    # st.write(selected_points)
                    if selected_points:
                        x = selected_points[0]['x']
                        y = selected_points[0]['y']-45
                        # st.write(x)
                        # st.write(y)
                        sd = df[(-df['LOC_X'] == x) & (df['LOC_Y'] == y)]
                        # st.write(sd)
                    
                        headers = {
                            'Host': 'stats.nba.com',
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
                            'Accept': 'application/json, text/plain, */*',
                            'Accept-Language': 'en-US,en;q=0.5',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'x-nba-stats-origin': 'stats',
                            'x-nba-stats-token': 'true',
                            'Connection': 'keep-alive',
                            'Referer': 'https://stats.nba.com/',
                            'Pragma': 'no-cache',
                            'Cache-Control': 'no-cache'
                        }
                        event_id = sd['GAME_EVENT_ID'].iloc[0]
                        game_id = sd['GAME_ID'].iloc[0]
                        event = sd['EVENT_TYPE'].iloc[0]
                        action = sd['ACTION_TYPE'].iloc[0]
                        sub = f'{event} {action}'
                    
                        url = 'https://stats.nba.com/stats/videoeventsasset?GameEventID={}&GameID={}'.format(
                                    event_id, game_id)
                        r = requests.get(url, headers=headers)
                        if r.status_code == 200:
                            json = r.json()
                            video_urls = json['resultSets']['Meta']['videoUrls']
                            playlist = json['resultSets']['playlist']
                            video_event = {'video': video_urls[0]['lurl'], 'desc': playlist[0]['dsc']}
                            video = video_urls[0]['lurl']
                        # col1,col2,col3 = st.columns(3)
                        # with col2:
                        st.video(video,autoplay=True)
    
                else:    
                    st.plotly_chart(fig)
        
            
            # Assuming df has a 'GAME_DATE' column and a 'SHOT_MADE_FLAG' column
            c1,c2 = st.columns(2)
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%Y%m%d')
            
            # Aggregate shooting data by date
            shooting_over_time = df.groupby('GAME_DATE').agg({"SHOT_MADE_FLAG": ["sum", "count"]}).reset_index()
            shooting_over_time.columns = ['Date', 'Made', 'Total']
            
            # Calculate shooting percentage
            shooting_over_time['Percentage'] = round(shooting_over_time['Made'] / shooting_over_time['Total'] * 100,2)
            
            # Add a moving average to smooth out trends (optional)
            shooting_over_time['Moving Average (7 Days)'] = round(shooting_over_time['Percentage'].rolling(window=7, min_periods=1).mean(),2)
            
            # Create the line chart
            fig2 = px.line(
                shooting_over_time, 
                x='Date', 
                y='Percentage', 
                title="Shooting Percentage Over Time",
                labels={'Percentage': 'Shooting Percentage (%)', 'Date': 'Game Date'},  # Axis labels
                markers=True,  # Show markers for each data point
                hover_data={'Date': True, 'Percentage': True, 'Made': True, 'Total': True},  # Show extra info on hover
            )
            
            # Add a moving average line (optional)
            fig2.add_scatter(
                x=shooting_over_time['Date'], 
                y=shooting_over_time['Moving Average (7 Days)'], 
                mode='lines', 
                name='7-Day Moving Average',
                line=dict(color='red', dash='dash')
            )
            
            # Customize the layout for better aesthetics
            fig2.update_layout(
                title="Shooting Percentage Over Time",
                title_x=0,  # Center the title
                title_font=dict(size=20, family='Arial', color='white'),
                xaxis=dict(
                    showgrid=True, 
                    tickangle=45,  # Rotate x-axis labels for better readability
                    tickformat='%b %d, %Y',  # Show date in a readable format
                    title_font=dict(size=14, family='Arial', color='white'),
                ),
                yaxis=dict(
                    title="Shooting Percentage (%)",
                    title_font=dict(size=14, family='Arial', color='white'),
                ),
                showlegend=False,  # Show the legend for the moving average line
                margin=dict(l=40, r=40, t=50, b=40),  # Adjust margins for better spacing
                plot_bgcolor='rgba(0, 0, 0, 0)',  # Set background color to white for a clean look
            )
            with c1:
                st.plotly_chart(fig2)
            
            # Create distance bins
            distance_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # adjusted max distance
            df_filtered = df.dropna(subset=['SHOT_DISTANCE'])

            df_filtered['distance_bin'] = pd.cut(df_filtered['SHOT_DISTANCE'], bins=distance_bins)
            
            # Convert the 'distance_bin' to strings to make it serializable
            df_filtered['distance_bin'] = df_filtered['distance_bin'].astype(str)
            
            # Calculate shooting percentage by distance bin
            shooting_by_distance = df_filtered.groupby('distance_bin').agg({"SHOT_MADE_FLAG": ["sum", "count"]}).reset_index()
            
            # Flatten the MultiIndex columns
            shooting_by_distance.columns = ['Distance', 'Made', 'Total']
            
            # Calculate accuracy
            shooting_by_distance['Percentage'] = round(shooting_by_distance['Made'] / shooting_by_distance['Total'] * 100,2)
            
            # Plot the figure using Plotly with a new theme
            fig3 = px.bar(
                shooting_by_distance, 
                x='Percentage', 
                y='Distance', 
                orientation='h',  # horizontal bars
                title="Shooting Percentage by Distance",
                color='Percentage',  # Use color to represent shooting percentage
                color_continuous_scale='YlOrRd',  # A warm color palette (yellow to red)
                labels={'Percentage': 'Shooting %', 'Distance': 'Shot Distance (ft)'},
                text='Percentage',  # Show percentage on the bars
            )
            fig3.update_coloraxes(showscale=False) 
            
            # Customize the layout for better aesthetics
            fig3.update_layout(
                title_text="Shooting Percentage by Distance",
                title_x=0,  # Center the title
                title_font=dict(size=20, family='Arial', color='white'),
                xaxis_title='Shooting Percentage (%)',
                yaxis_title='Shot Distance (ft)',
                yaxis_tickangle=-45,  # Rotate y-axis labels for readability
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                plot_bgcolor='rgba(0, 0, 0, 0)', # Set background color to white for clarity
                margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins to fit content
                showlegend=False  # Disable legend for this chart
            )
            
            # Add annotations to show the percentage on each bar
            fig3.update_traces(
                texttemplate='%{text:.1f}%',  # Format percentage to 1 decimal place
                textposition='inside',  # Position the percentage text inside the bars
                insidetextanchor='middle'
            )
            
            # Display the plot in Streamlit
            with c2:
                st.plotly_chart(fig3)
            
            # Assuming df has a column 'ACTION_TYPE' for shot type and 'SHOT_MADE_FLAG' for success
            co1, co2 = st.columns(2)
            shot_accuracy_by_type = df.groupby('ACTION_TYPE').agg({"SHOT_MADE_FLAG": ["sum", "count"]}).reset_index()
            
            # Flatten MultiIndex columns
            shot_accuracy_by_type.columns = ['Shot Type', 'Made', 'Total']
            
            # Calculate shooting percentage
            shot_accuracy_by_type['Percentage'] = shot_accuracy_by_type['Made'] / shot_accuracy_by_type['Total'] * 100
            
            # Create a bar plot with enhancements
            fig = px.bar(
                shot_accuracy_by_type, 
                x='Percentage', 
                y='Shot Type', 
                orientation='h',  # Horizontal bars for better readability
                title="Shooting Percentage by Shot Type",
                color='Percentage',  # Color bars based on shooting percentage
                color_continuous_scale='Viridis',  # Modern color scale (Viridis)
                labels={'Percentage': 'Shooting %', 'Shot Type': 'Shot Type'},
                text='Shot Type',  # Show percentage on bars
            )
            fig.update_coloraxes(showscale=False) 


            
            # Customize the layout to improve aesthetics
            fig.update_layout(
                title_text="Shooting Percentage by Shot Type",
                title_x=0,  # Center the title
                title_font=dict(size=20, family='Arial', color='white'),
                xaxis_title='Shooting Percentage (%)',
                yaxis_title='Shot Type',
                yaxis_tickangle=-45,  # Rotate y-axis labels for better readability
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                plot_bgcolor='rgba(0, 0, 0, 0)',  # Set background color to white
                margin=dict(l=20, r=20, t=40, b=40),  # Adjust margins for better fit
                showlegend=False,  # Disable legend for clarity
                bargap=0.1,  # Set the gap between bars (lower value = wider bars)
                bargroupgap=0.05,  # Set gap between bar groups (useful for grouped bars)
            )
            
            # Update traces to show percentage inside bars
            fig.update_traces(
                # texttemplate='%{text:.1f}%',  # Format text to 1 decimal place
                textposition='inside',  # Position the text inside the bars
                insidetextanchor='middle'  # Align text in the middle of bars
            )
            fig.update_layout(
                yaxis=dict(
                    showticklabels=False,  # Hide the tick labels
                    zeroline=False,        # Optionally, remove the zero line if you want a cleaner look
                    showline=False         # Optionally, remove the axis line
                )
            )
            
            # Add hover information to show shot type, made count, and total shots
            fig.update_traces(
                hovertemplate='%{y}<br>%{customdata[0]}/%{customdata[1]}<br>%{x:.1f}%',
                customdata=shot_accuracy_by_type[['Made', 'Total']].values
            )
            with co1:
                st.plotly_chart(fig)
            if fgperc:
         
                x_bins = np.linspace(-270, 270, 30)  # 30 bins along X axis (basketball court length)
                y_bins = np.linspace(-10, 450, 20)   # 20 bins along Y axis (basketball court width)
                
                # Create 2D histograms: one for shot attempts (total shots) and one for made shots
                shot_attempts, x_edges, y_edges = np.histogram2d(df['LOC_X'], df['LOC_Y'], bins=[x_bins, y_bins])
                made_shots, _, _ = np.histogram2d(df['LOC_X'][df['SHOT_MADE_FLAG'] == 1], df['LOC_Y'][df['SHOT_MADE_FLAG'] == 1], bins=[x_bins, y_bins])
                
                # Calculate the Field Goal Percentage (FG%) for each bin
                fg_percentage = np.divide(made_shots, shot_attempts, where=shot_attempts != 0) * 100  # Avoid division by zero
                
                # Normalize FG% for color mapping (to make sure it stays between 0 and 100)
                fg_percentage_normalized = np.clip(fg_percentage, 0, 100)  # Clamp FG% between 0 and 100
                
                # Calculate the center of each bin for plotting (bin centers)
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                
                # Create a meshgrid of X and Y centers for 3D plotting
                X, Y = np.meshgrid(x_centers, y_centers)
                
                # Create hovertext to show FG% for each region
                hovertext = np.array([f'FG%: {fg}%' for fg in fg_percentage.flatten()]).reshape(fg_percentage.shape)

                
                # Create the 3D surface plot
                z_max = 100  # Replace with the desired limit
                Z = shot_attempts.T
                Z2 = Z *5
                Z2 = np.minimum(Z2, z_max)
                fig = go.Figure(data=go.Surface(
                    z=Z2,  # Shot density (number of shots) as the Z-axis
                    x=-X,  # X values (bin centers)
                    y=Y+45,  # Y values (bin centers)
                    
                    # Surface color based on Field Goal Percentage (FG%)
                    surfacecolor=fg_percentage.T,  # Use FG% as the surface color
                    
                    colorscale='hot',  # Color scale based on FG% (you can change this to any scale)
                    cmin=0,  # Minimum FG% for color scale
                    cmax=100,  # Maximum FG% for color scale
                    colorbar=dict(title='Field Goal %'),  # Color bar label
                    showscale=False,  # Show the color scale/legend
                    hoverinfo='none',  # Show text on hover
                    # hovertext=hovertext  # Attach the hovertext showing FG%
                ))
            else:
                x_bins = np.linspace(-270, 270, 30)  # 30 bins along X axis (basketball court length)
                y_bins = np.linspace(-10, 450, 20)  # 20 bins along Y axis (basketball court width)
                
                # Create 2D histogram to get shot density
                shot_density, x_edges, y_edges = np.histogram2d(df['LOC_X'], df['LOC_Y'], bins=[x_bins, y_bins])
                
                # Calculate the center of each bin for plotting
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                
                # Create a meshgrid of X and Y centers for 3D plotting
                X, Y = np.meshgrid(x_centers, y_centers)
                Z = shot_density.T  # Transpose to match the correct orientation for plotting
                Z2 = Z*5
                z_max = 100  # Replace with the desired limit
    
                # Apply the limit to Z values
                Z2 = np.minimum(Z2, z_max)
                # Plot 3D shot density
                hovertext = np.array([f'Shots: {z}' for z in Z.flatten()]).reshape(Z.shape)
                fig = go.Figure(data=go.Surface(
                    z=Z2,
                    x=-X,
                    y=Y+45,
                    colorscale='hot',  # You can choose different color scales
                    colorbar=dict(title='Shot Density'),
                    showscale=False  # Hide the color bar/legend
                    ,hoverinfo='text',
                    hovertext=hovertext
                ))
            court_perimeter_lines = court_lines_df[court_lines_df['line_id'] == 'outside_perimeter']
            three_point_lines = court_lines_df[court_lines_df['line_id'] == 'three_point_line']
            backboard = court_lines_df[court_lines_df['line_id'] == 'backboard']
            backboard2 = court_lines_df[court_lines_df['line_id'] == 'backboard2']
            freethrow = court_lines_df[court_lines_df['line_id'] == 'free_throw_line']
            freethrow2 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line2']
            freethrow3 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line3']
            freethrow4 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line4']
            freethrow5 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line5']
            hoop = court_lines_df[court_lines_df['line_id'] == 'hoop']
            hoop2 = court_lines_df[court_lines_df['line_id'] == 'hoop2']
            
            
            
            
            
            
            
            # Add court lines to the plot (3D scatter)
            fig.add_trace(go.Scatter3d(
                x=court_perimeter_lines['x'],
                y=court_perimeter_lines['y'],
                z=np.zeros(len(court_perimeter_lines)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='white', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=hoop['x'],
                y=hoop['y'],
                z=hoop['z'],  # Place 3-point line on the floor
                mode='lines',
                line=dict(color='#e47041', width=6),
                name="Hoop",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=hoop2['x'],
                y=hoop2['y'],
                z=hoop2['z'],  # Place 3-point line on the floor
                mode='lines',
                line=dict(color='#D3D3D3', width=6),
                name="Backboard",
                hoverinfo='none'
            ))
            # Add the 3-point line to the plot
            fig.add_trace(go.Scatter3d(
                x=backboard['x'],
                y=backboard['y'],
                z=backboard['z'],  # Place 3-point line on the floor
                mode='lines',
                line=dict(color='grey', width=6),
                name="Backboard",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=backboard2['x'],
                y=backboard2['y'],
                z=backboard2['z'],  # Place 3-point line on the floor
                mode='lines',
                line=dict(color='grey', width=6),
                name="Backboard",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=three_point_lines['x'],
                y=three_point_lines['y'],
                z=np.zeros(len(three_point_lines)),  # Place 3-point line on the floor
                mode='lines',
                line=dict(color='white', width=6),
                name="3-Point Line",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=freethrow['x'],
                y=freethrow['y'],
                z=np.zeros(len(freethrow)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='white', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=freethrow2['x'],
                y=freethrow2['y'],
                z=np.zeros(len(freethrow2)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='white', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=freethrow3['x'],
                y=freethrow3['y'],
                z=np.zeros(len(freethrow3)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='white', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=freethrow4['x'],
                y=freethrow4['y'],
                z=np.zeros(len(freethrow4)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='white', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=freethrow5['x'],
                y=freethrow5['y'],
                z=np.zeros(len(freethrow5)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='white', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            # Update layout for better visuals
            if vids:
                h = 470
                bgcolor = 'white'
                showbg = True
            else:
                h = 600
                bgcolor = 'white'
                showbg = False
            court_perimeter_bounds = np.array([[-250, 0, -0.2], [250, 0, -0.2], [250, 450, -0.2], [-250, 450, -0.2], [-250, 0, -0.2]])

            # Extract x, y, and z values for the mesh
            court_x = court_perimeter_bounds[:, 0]
            court_y = court_perimeter_bounds[:, 1]
            court_z = court_perimeter_bounds[:, 2]
            
            # Add a square mesh to represent the court floor at z=0
            fig.add_trace(go.Mesh3d(
                x=court_x,
                y=court_y,
                z=court_z,
                color='black',
                # opacity=0.5,
                name='Court Floor',
                hoverinfo='none',
                showscale=False
            ))
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                scene_aspectmode="data",
                height=h,
                scene_camera=dict(
                    eye=dict(x=1.3, y=0, z=0.7)
                ),
                title="",
                scene=dict(
                     xaxis=dict(title='', showticklabels=False, showgrid=False,showbackground=showbg,backgroundcolor=bgcolor),
                        yaxis=dict(title='', showticklabels=False, showgrid=False,showbackground=showbg,backgroundcolor=bgcolor),
                        zaxis=dict(title='',  showticklabels=False, showgrid=False,showbackground=False,backgroundcolor='black'),
               
            ),
             showlegend=False
            )
            
            # Show the plot in Streamlit
            with coli2:
                st.plotly_chart(fig,use_container_width=True)
            
            shot_type_distribution = df['ACTION_TYPE'].value_counts().reset_index()
            shot_type_distribution.columns = ['Shot Type', 'Count']
            
            # Create a pie chart with enhancements
            fig = px.pie(
                shot_type_distribution, 
                names='Shot Type', 
                values='Count', 
                # title="Shot Type Distribution", 
                color='Shot Type',  # Color by shot type for better distinction
                color_discrete_sequence=px.colors.sequential.Plasma,  # Modern color scale
                hole=0.3,  # Create a donut chart (optional)
                # hover_data={'Shot Type': False, 'Count': True},  # Hover will show only 'Count', not 'Shot Type'
            )
            
            # Customize layout for better visual appeal
            fig.update_layout(
                title="Shot Type Distribution",
                title_x=0,  # Center the title
                title_y=1,
                title_font=dict(size=20, family='Arial', color='white'),
                showlegend=False,  # Show the legend
                legend_title='Shot Type',
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="center", 
                    x=0.5
                ),
                margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins
                plot_bgcolor='rgba(0, 0, 0, 0)',  # Set background color to white for a clean look
            )
            
            # Enhance hover info and show percentages on slices
            fig.update_traces(
                textinfo='percent+label',  # Display percentage and shot type
                pull=[0.1 if i == shot_type_distribution['Count'].idxmax() else 0 for i in range(len(shot_type_distribution))],  # Explode the max slice
                hovertemplate='Shot Type: %{label}<br>Count: %{value}'  # Detailed hover info
            )
            
            with co2:
                st.plotly_chart(fig)
        else:
            st.error('No data found')
            
            
            
             
