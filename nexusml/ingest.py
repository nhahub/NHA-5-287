"""Normalise an uploaded table (CSV, Excel, TSV, JSON / JSON-lines, XML, Parquet) to CSV bytes so the rest of the pipeline only ever reads CSV."""
import io


def to_csv_bytes(name: str, raw: bytes) -> bytes:
    n = name.lower()
    if n.endswith((".xlsx", ".xlsm", ".xls")):
        import pandas as pd
        try:
            df = pd.read_excel(io.BytesIO(raw))                  # first sheet; needs openpyxl (.xlsx) / xlrd (.xls)
        except ImportError as e:
            raise ValueError("Reading Excel files needs the 'openpyxl' package (pip install openpyxl).") from e
        return df.to_csv(index=False).encode("utf-8")
    if n.endswith((".tsv", ".tab")):
        import pandas as pd
        return pd.read_csv(io.BytesIO(raw), sep="\t").to_csv(index=False).encode("utf-8")
    if n.endswith((".json", ".jsonl", ".ndjson")):
        import json
        import pandas as pd
        text = raw.decode("utf-8-sig")
        try:
            if n.endswith(".json"):
                data = json.loads(text)
                if isinstance(data, dict):                                  # {"data": [...]} / {"records": [...]} / one big dict of lists
                    data = next((v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)), data)
                df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame(data)
            else:
                df = pd.json_normalize([json.loads(l) for l in text.splitlines() if l.strip()])
        except Exception as e:
            raise ValueError(f"Could not turn that JSON into a table: {e}") from e
        return df.to_csv(index=False).encode("utf-8")
    if n.endswith(".xml"):
        import pandas as pd
        try:
            return pd.read_xml(io.BytesIO(raw)).to_csv(index=False).encode("utf-8")
        except Exception as e:
            raise ValueError(f"Could not turn that XML into a table (it needs repeated records): {e}") from e
    if n.endswith(".parquet"):
        import pandas as pd
        try:
            return pd.read_parquet(io.BytesIO(raw)).to_csv(index=False).encode("utf-8")
        except ImportError as e:
            raise ValueError("Reading Parquet needs 'pyarrow' (pip install pyarrow).") from e
    return raw
