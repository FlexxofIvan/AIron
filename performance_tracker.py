import json
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

class PerformanceTracker:
    def __init__(self, db_path: str = "fitness_data.db"):
        self.db_path = db_path
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize SQLite database for workout tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                duration_minutes REAL,
                total_reps INTEGER,
                avg_form_score REAL,
                exercises_data TEXT
            )
        ''')
        
        # Create exercises table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                exercise_type TEXT,
                reps INTEGER,
                form_scores TEXT,
                mistakes TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        # Create form_data table for detailed tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS form_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                exercise_type TEXT,
                timestamp TEXT,
                form_score REAL,
                knee_angle REAL,
                elbow_angle REAL,
                biomech_data TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_session_to_db(self, session_data: Dict) -> int:
        """Save complete session to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert session record
        cursor.execute('''
            INSERT INTO sessions (date, duration_minutes, total_reps, avg_form_score, exercises_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            session_data.get('duration_minutes', 0),
            session_data.get('total_reps', 0),
            session_data.get('avg_form_score', 0),
            json.dumps(session_data.get('exercises', {}))
        ))
        
        session_id = cursor.lastrowid
        
        # Insert exercise records
        for exercise_type, exercise_data in session_data.get('exercises', {}).items():
            cursor.execute('''
                INSERT INTO exercises (session_id, exercise_type, reps, form_scores, mistakes)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                session_id,
                exercise_type,
                exercise_data.get('reps', 0),
                json.dumps(exercise_data.get('form_scores', [])),
                json.dumps(exercise_data.get('common_mistakes', []))
            ))
        
        conn.commit()
        conn.close()
        return session_id
    
    def get_workout_history(self, days: int = 30) -> pd.DataFrame:
        """Get workout history from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM sessions 
            WHERE date >= ? 
            ORDER BY date DESC
        '''
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        df = pd.read_sql_query(query, conn, params=[cutoff_date])
        conn.close()
        
        return df
    
    def create_progress_charts(self, days: int = 30) -> Dict[str, go.Figure]:
        """Create interactive progress charts using Plotly"""
        df = self.get_workout_history(days)
        charts = {}
        
        if df.empty:
            # Return empty charts
            fig_empty = go.Figure()
            fig_empty.add_annotation(text="No workout data available", 
                                   xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return {'progress': fig_empty, 'form_trend': fig_empty, 'volume': fig_empty}
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. Form score progress over time
        fig_progress = px.line(df, x='date', y='avg_form_score', 
                              title='Form Score Progress Over Time',
                              labels={'avg_form_score': 'Average Form Score (%)', 'date': 'Date'})
        fig_progress.add_hline(y=80, line_dash="dash", line_color="green", 
                              annotation_text="Excellent (80%)")
        fig_progress.add_hline(y=60, line_dash="dash", line_color="orange", 
                              annotation_text="Good (60%)")
        charts['progress'] = fig_progress
        
        # 2. Workout volume trend
        fig_volume = px.bar(df, x='date', y='total_reps',
                           title='Workout Volume Trend',
                           labels={'total_reps': 'Total Reps', 'date': 'Date'})
        charts['volume'] = fig_volume
        
        # 3. Form score distribution
        fig_distribution = px.histogram(df, x='avg_form_score', nbins=20,
                                       title='Form Score Distribution',
                                       labels={'avg_form_score': 'Form Score (%)', 'count': 'Frequency'})
        charts['distribution'] = fig_distribution
        
        return charts
    
    def create_exercise_comparison_chart(self) -> go.Figure:
        """Create comparison chart across different exercises"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT exercise_type, AVG(reps) as avg_reps, form_scores
            FROM exercises
            GROUP BY exercise_type
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No exercise data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate average form scores for each exercise
        avg_form_scores = []
        for _, row in df.iterrows():
            scores = json.loads(row['form_scores']) if row['form_scores'] else []
            avg_score = np.mean(scores) if scores else 0
            avg_form_scores.append(avg_score)
        
        df['avg_form_score'] = avg_form_scores
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Average Reps',
            x=df['exercise_type'],
            y=df['avg_reps'],
            yaxis='y',
            offsetgroup=1
        ))
        
        fig.add_trace(go.Bar(
            name='Average Form Score',
            x=df['exercise_type'],
            y=df['avg_form_score'],
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig.update_layout(
            title='Exercise Performance Comparison',
            xaxis_title='Exercise Type',
            yaxis=dict(title='Average Reps', side='left'),
            yaxis2=dict(title='Average Form Score (%)', side='right', overlaying='y'),
            barmode='group'
        )
        
        return fig
    
    def get_personal_records(self) -> Dict:
        """Get personal records and achievements"""
        conn = sqlite3.connect(self.db_path)
        
        # Best form scores
        query_best_form = '''
            SELECT exercise_type, MAX(avg_form_score) as best_form
            FROM (
                SELECT session_id, exercise_type, 
                       AVG(CAST(json_extract(form_scores, '$[' || (ROW_NUMBER() OVER () - 1) || ']') AS REAL)) as avg_form_score
                FROM exercises
                GROUP BY session_id, exercise_type
            )
            GROUP BY exercise_type
        '''
        
        # Most reps in session
        query_most_reps = '''
            SELECT exercise_type, MAX(reps) as max_reps
            FROM exercises
            GROUP BY exercise_type
        '''
        
        best_form_df = pd.read_sql_query(query_best_form, conn)
        most_reps_df = pd.read_sql_query(query_most_reps, conn)
        
        conn.close()
        
        records = {
            'best_form_scores': best_form_df.to_dict('records') if not best_form_df.empty else [],
            'max_reps': most_reps_df.to_dict('records') if not most_reps_df.empty else [],
            'total_sessions': len(self.get_workout_history(365)),
            'total_workouts': self.get_workout_history(365)['total_reps'].sum() if not self.get_workout_history(365).empty else 0
        }
        
        return records
    
    def create_achievement_system(self, session_data: Dict, personal_records: Dict) -> List[str]:
        """Create achievement/badge system"""
        achievements = []
        
        current_form = session_data.get('avg_form_score', 0)
        current_reps = session_data.get('total_reps', 0)
        
        # Form-based achievements
        if current_form >= 90:
            achievements.append("🏆 Perfect Form Master")
        elif current_form >= 80:
            achievements.append("🥇 Form Expert")
        elif current_form >= 70:
            achievements.append("🥈 Good Form")
        
        # Rep-based achievements
        if current_reps >= 100:
            achievements.append("💪 Century Club")
        elif current_reps >= 50:
            achievements.append("🔥 Half Century")
        elif current_reps >= 25:
            achievements.append("⭐ Quarter Century")
        
        # Consistency achievements
        history_df = self.get_workout_history(7)  # Last 7 days
        if len(history_df) >= 7:
            achievements.append("📅 7-Day Streak")
        elif len(history_df) >= 3:
            achievements.append("🎯 3-Day Streak")
        
        # Improvement achievements
        if len(history_df) >= 2:
            recent_scores = history_df['avg_form_score'].tolist()
            if len(recent_scores) >= 2 and recent_scores[0] > recent_scores[-1] + 10:
                achievements.append("📈 Rapid Improvement")
        
        return achievements