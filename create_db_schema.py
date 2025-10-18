#!/usr/bin/env python3
"""
Database schema creation script for analysis-service
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def create_database_schema():
    """Create all required tables for analysis-service"""
    
    # Database connection parameters
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    database = os.getenv("DB_NAME", "mong-analysis")
    
    # Create database URL
    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    
    print(f"Connecting to database: {host}:{port}/{database}")
    
    try:
        # Create engine
        engine = create_engine(db_url, echo=True)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
        
        # Import all models to ensure they are registered with SQLAlchemy
        from app.api.domain.domain.entity.sleep_session_entity import (
            SleepSession, DailyReport, SleepTimeDetail, AnalysisDetail, AnalysisStep
        )
        from app.api.domain.domain.entity.periodic_report_entity import (
            PeriodicReport, ScorePredictionPoint
        )
        from app.api.domain.domain.entity.analyzed_data_entity import (
            SoundEvent, SleepLevel
        )
        
        print("✅ All models imported successfully!")
        
        # Create all tables
        from app.api.domain.domain.entity.base import Base
        Base.metadata.create_all(bind=engine)
        
        print("✅ All tables created successfully!")
        
        # Show created tables
        with engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            print(f"\n📋 Created tables ({len(tables)}):")
            for table in sorted(tables):
                print(f"  - {table}")
        
        return True
        
    except SQLAlchemyError as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating analysis-service database schema...")
    print("=" * 50)
    
    success = create_database_schema()
    
    if success:
        print("\n🎉 Database schema creation completed successfully!")
    else:
        print("\n💥 Database schema creation failed!")
        sys.exit(1)
