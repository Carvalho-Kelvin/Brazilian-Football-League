import gspread, pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

_SCOPE = [ "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",]

_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ZKm79JbSoFn7ljR7fGQci4uVAx2p5sMXel6hD-ZgQHs"


def load_matches():
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "brazilian-football.json", _SCOPE
        )
        ws = gspread.authorize(creds).open_by_url(_SHEET_URL).sheet1
        return pd.DataFrame(ws.get_all_records())
    except FileNotFoundError:
        print("Error: credential file 'brazilian-football.json' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading matches: {e}")
        return pd.DataFrame()