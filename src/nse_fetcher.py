import io
import zipfile
import requests
import pandas as pd
from datetime import datetime

NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br'
}

def get_nse_session() -> requests.Session:
    """Establishes an active session with valid NSE cookies."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Pre-flight request to get session cookies
        session.get('https://www.nseindia.com', timeout=10)
    except Exception as e:
        print(f"Warning: Could not establish initial NSE session: {e}")
    return session

def fetch_single_bhavcopy(date_obj: datetime) -> pd.DataFrame:
    """Downloads and cleans NSE Bhavcopy for a given business day."""
    date_str = date_obj.strftime('%d%b%Y').upper()
    year_str = date_obj.strftime('%Y')
    month_str = date_obj.strftime('%b').upper()
    
    url = f"https://archives.nseindia.com/content/historical/EQUITIES/{year_str}/{month_str}/cm{date_str}bhav.csv.zip"
    
    try:
        res = requests.get(url, headers=NSE_HEADERS, timeout=10)
        if res.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f)
                    df = df[df['SERIES'] == 'EQ'].copy()
                    df['DATE'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%b-%Y')
                    return df[['DATE', 'SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY', 'TOTTRDVAL']]
    except Exception:
        pass
    return None

def fetch_nse_corporate_actions(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches official corporate action data (Splits, Bonuses, Rights, Demergers)
    directly from NSE API.
    """
    session = get_nse_session()
    # Format dates as DD-MM-YYYY for NSE API
    s_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%d-%m-%Y')
    e_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%d-%m-%Y')
    
    url = f"https://www.nseindia.com/api/corporates-corporateActions?index=equities&from={s_date}&to={e_date}"
    
    try:
        res = session.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if data:
                df = pd.DataFrame(data)
                # Keep critical corporate action columns
                df = df[['symbol', 'exDate', 'subject']].copy()
                df.columns = ['SYMBOL', 'EX_DATE', 'SUBJECT']
                df['EX_DATE'] = pd.to_datetime(df['EX_DATE'], format='%d-%b-%Y', errors='coerce')
                return df.dropna(subset=['EX_DATE'])
    except Exception as e:
        print(f"Error fetching NSE Corporate Actions: {e}")
        
    return pd.DataFrame(columns=['SYMBOL', 'EX_DATE', 'SUBJECT'])