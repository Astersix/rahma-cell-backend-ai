import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
import pandas as pd
from filelock import FileLock
from pathlib import Path

# Import training functions dari train.py (exact same logic)
from train import train_variant, MIN_DATA_POINTS

# File locking configuration
LOCK_FILE = "models/.retrain.lock"
LOCK_TIMEOUT = 3600  # 1 hour timeout

load_dotenv()

# setup orm
engine = create_engine(os.getenv('DATABASE_URL'))
Base = automap_base()
Base.prepare(engine, reflect=True)

# akses ke tabel
order = Base.classes.order
order_product = Base.classes.order_product

def fetch_training_data(product_variant_id=None):
    with Session(engine) as session:
        query = session.query(
            order_product.product_variant_id,
            order.created_at.label('date'),
            order_product.quantity,
            order_product.price,
        ).join(order, order.id == order_product.order_id).filter(order.status == 'completed')

        if product_variant_id:
            query = query.filter(order_product.product_variant_id == product_variant_id)
        
        query = query.order_by(order.created_at.asc())

        # ubah ke dataframe
        df = pd.read_sql(query.statement, session.bind)
        
        # Pastikan date dalam format datetime
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"fetched {len(df)} records from database")
        return df

def retrain_all_models():
    print("retraining models from db")
    
    # Acquire file lock to prevent concurrent retraining
    lock = FileLock(LOCK_FILE, timeout=LOCK_TIMEOUT)
    try:
        lock.acquire()
        print("lock acquired - starting retraining...")
    except Exception as e:
        print(f"error: could not acquire lock - {str(e)}")
        print("another retraining process is already running")
        return
    
    try:
        # Fetch all data dari database
        try:
            df_all = fetch_training_data()
        except Exception as e:
            print(f"Error: failed to fetch data from database - {str(e)}")
            return
        
        if len(df_all) == 0:
            print("error: no data found in database")
            print("make sure there are orders with status='completed'")
            return
        
        print(f"total records: {len(df_all)}")
        
        # Get unique variants
        unique_variants = df_all['product_variant_id'].unique()
        print(f"\nfound {len(unique_variants)} unique product variants")
        
        # Train untuk setiap variant
        for i, variant_id in enumerate(unique_variants):
            print(f"\n[{i+1}/{len(unique_variants)}] processing {variant_id}")
            df_variant = df_all[df_all['product_variant_id'] == variant_id].sort_values('date')
            
            try:
                train_variant(df_variant, variant_id)
            except Exception as e:
                print(f"Failed: {str(e)}")
        
        print("retraining completed")
    finally:
        # Always release lock
        lock.release()
        print("lock released")

if __name__ == '__main__':
    retrain_all_models()