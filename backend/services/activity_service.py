"""Small local JSON activity feed for the dashboard."""
from datetime import datetime,timezone
from backend.config import REPORTS
from backend.utils.helpers import read_json,write_json

def record_activity(kind,message,details=None):
    path=REPORTS/'activities.json'; items=read_json(path,[]) or []
    item={'id':datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f'),'kind':kind,'message':message,'details':details or {},'timestamp':datetime.now(timezone.utc).isoformat()}
    items.insert(0,item); write_json(items[:30],path); return item

def recent_activities(limit=8): return (read_json(REPORTS/'activities.json',[]) or [])[:limit]
