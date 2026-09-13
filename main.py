import os
import io
import zipfile
import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
import aiohttp
import numpy as np

# ====================== AYARLAR ======================
TIMEFRAME = '15m'
RSI_PERIOD = 14
MIN_BARS = 6
MAX_BARS = 80

MIN_PRICE_PCT = 0.50          # Fiyat farkı min %
MIN_RSI_DIFF = 3.0            # RSI farkı min
MIN_SWING_ATR = 1.20          # Swing min ATR çarpanı
MIN_SWING_PCT = 1.00          # Swing min %
MIN_NEW_EXTREME_PCT = 0.08    # Canlı 2. dip/tepe önceki mumların dışına taşmalı (%)

# RSI bölge filtreleri (çok önemli)
BULL_FIRST_RSI_MAX = 42.0
BULL_SECOND_RSI_MAX = 50.0
BEAR_FIRST_RSI_MIN = 58.0
BEAR_SECOND_RSI_MIN = 50.0

ATR_PERIOD = 14
MAX_CANDLES = 240
HISTORY_DAYS = 3
COOLDOWN = 1800               # saniye

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHAT = os.getenv('TELEGRAM_CHAT_ID', '')

# ====================== SEMBOL LİSTESİ ======================
SYMBOLS = '''BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT DOTUSDT TRXUSDT LTCUSDT BCHUSDT SUIUSDT HBARUSDT XLMUSDT TONUSDT SHIBUSDT UNIUSDT AAVEUSDT NEARUSDT APTUSDT ICPUSDT ETCUSDT FILUSDT ATOMUSDT ALGOUSDT VETUSDT MKRUSDT ARBUSDT OPUSDT INJUSDT STXUSDT IMXUSDT SEIUSDT TIAUSDT GRTUSDT RUNEUSDT EGLDUSDT SANDUSDT MANAUSDT AXSUSDT THETAUSDT FLOWUSDT XTZUSDT EOSUSDT NEOUSDT KAVAUSDT SNXUSDT CRVUSDT DYDXUSDT 1INCHUSDT COMPUSDT ENSUSDT LDOUSDT RLCUSDT CHZUSDT GALAUSDT APEUSDT WOOUSDT ZILUSDT ENJUSDT IOTAUSDT KSMUSDT CELOUSDT ROSEUSDT ANKRUSDT SKLUSDT ZRXUSDT QTUMUSDT ONTUSDT ONEUSDT DASHUSDT YFIUSDT SUSHIUSDT UMAUSDT API3USDT GMTUSDT MASKUSDT ASTRUSDT JASMYUSDT LPTUSDT MINAUSDT CFXUSDT IDUSDT EDUUSDT SFPUSDT DUSKUSDT BLURUSDT HIGHUSDT HOOKUSDT ARPAUSDT CTSIUSDT CELRUSDT LQTYUSDT RDNTUSDT STGUSDT TRBUSDT SSVUSDT CYBERUSDT WLDUSDT ORDIUSDT BOMEUSDT MEMEUSDT WIFUSDT JTOUSDT JUPUSDT PYTHUSDT DYMUSDT PORTALUSDT PIXELUSDT STRKUSDT AEVOUSDT ENAUSDT ETHFIUSDT REZUSDT BBUSDT NOTUSDT ZKUSDT LISTAUSDT IOUSDT TNSRUSDT TAOUSDT OMNIUSDT MEWUSDT TURBOUSDT NEIROUSDT EIGENUSDT CATIUSDT HMSTRUSDT SCRUSDT GOATUSDT MOODENGUSDT ACTUSDT PNUTUSDT 1000PEPEUSDT 1000SHIBUSDT 1000BONKUSDT 1000FLOKIUSDT 1000SATSUSDT 1000LUNCUSDT SAGAUSDT MANTAUSDT ALTUSDT XAIUSDT ACEUSDT NTRNUSDT AIUSDT AIOZUSDT BIGTIMEUSDT COMBOUSDT DEGENUSDT DENTUSDT DODOXUSDT DOGSUSDT DRIFTUSDT FIDAUSDT FLMUSDT FIOUSDT FLUXUSDT FXSUSDT GHSTUSDT GLMUSDT GMXUSDT GRASSUSDT HIFIUSDT HFTUSDT HNTUSDT ILVUSDT JOEUSDT KASUSDT KDAUSDT KEYUSDT LEVERUSDT LOKAUSDT LUNA2USDT MAVUSDT MBOXUSDT METISUSDT MYROUSDT NFPUSDT NKNUSDT NMRUSDT OXTUSDT PENDLEUSDT PERPUSDT POLYXUSDT POWRUSDT PROMUSDT QNTUSDT RAREUSDT RENDERUSDT RONINUSDT RVNUSDT SANTOSUSDT SCRTUSDT SLERFUSDT SNTUSDT SPELLUSDT SUPERUSDT SXPUSDT TLMUSDT TOMOUSDT TRUUSDT USTCUSDT VOXELUSDT WAXPUSDT XMRUSDT YGGUSDT ZENUSDT ZETAUSDT ZROUSDT'''.split()

# ====================== GLOBAL STATE ======================
histories = {}
last_bull = {}
last_bear = {}
last_alert = {}
live_low = {}
live_high = {}

# ====================== GÖSTERGELER ======================
def rsi(close, n=14):
    c = pd.Series(close, dtype='float64')
    d = c.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = l.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()

def add_indicators(df):
    df = df.copy()
    df['rsi'] = rsi(df.close, RSI_PERIOD)
    pc = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - pc).abs(),
        (df.low - pc).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
    return df

# ====================== PİVOT ======================
def pivots(df, kind):
    """Daha sağlam pivot: 2 mum sol + 2 mum sağ"""
    a = df.low.to_numpy() if kind == 'low' else df.high.to_numpy()
    out = []
    for i in range(2, len(df) - 2):
        if kind == 'low':
            if (a[i] < a[i-1] and a[i] <= a[i+1] and
                a[i] < a[i-2] and a[i] <= a[i+2]):
                out.append(i)
        else:
            if (a[i] > a[i-1] and a[i] >= a[i+1] and
                a[i] > a[i-2] and a[i] >= a[i+2]):
                out.append(i)
    return out

# ====================== BULL SETUP (POZİTİF UYUMSUZLUK) ======================
def bull_setup(df):
    if len(df) < 25:
        return None

    ps = pivots(df, 'low')
    if not ps:
        return None

    # En son onaylanmış dipten başla
    for p in reversed(ps):
        bars = len(df) - 1 - p
        if bars < MIN_BARS or bars > MAX_BARS:
            continue

        op = float(df.iloc[p].low)
        orsi = float(df.iloc[p].rsi)
        atr = float(df.iloc[p].atr or 0)

        if atr <= 0 or np.isnan(orsi) or orsi > BULL_FIRST_RSI_MAX:
            continue

        # Aradaki swing (1. dip sonrası yukarı hareket)
        mid = df.iloc[p+1:-1]
        if len(mid) < MIN_BARS - 1:
            continue

        mid_high = float(mid.high.max())
        swing_pct = (mid_high / op - 1) * 100
        if swing_pct < MIN_SWING_PCT or (mid_high - op) < MIN_SWING_ATR * atr:
            continue

        # Canlı 2. dip
        nl = float(df.iloc[-1].low)
        nr = float(df.iloc[-1].rsi)
        if np.isnan(nr):
            continue

        # Canlı dip, son 3 kapalı mumun en düşüğünün de altına inmiş olmalı
        prior_low = float(df.iloc[-4:-1].low.min())
        if nl > prior_low * (1 - MIN_NEW_EXTREME_PCT / 100):
            continue

        # Asıl uyumsuzluk şartları
        price_diff = (nl / op - 1) * 100
        rsi_diff = nr - orsi

        if (price_diff <= -MIN_PRICE_PCT and
            rsi_diff >= MIN_RSI_DIFF and
            nr <= BULL_SECOND_RSI_MAX):

            return {
                'p': p,
                'time': str(df.iloc[p].open_time),
                'op': op,
                'np': nl,
                'orsi': orsi,
                'nr': nr,
                'pct': price_diff,
                'rd': rsi_diff,
                'bars': bars,
                'swing_pct': swing_pct
            }
    return None

# ====================== BEAR SETUP (NEGATİF UYUMSUZLUK) ======================
def bear_setup(df):
    if len(df) < 25:
        return None

    ps = pivots(df, 'high')
    if not ps:
        return None

    for p in reversed(ps):
        bars = len(df) - 1 - p
        if bars < MIN_BARS or bars > MAX_BARS:
            continue

        op = float(df.iloc[p].high)
        orsi = float(df.iloc[p].rsi)
        atr = float(df.iloc[p].atr or 0)

        if atr <= 0 or np.isnan(orsi) or orsi < BEAR_FIRST_RSI_MIN:
            continue

        mid = df.iloc[p+1:-1]
        if len(mid) < MIN_BARS - 1:
            continue

        mid_low = float(mid.low.min())
        swing_pct = (op / mid_low - 1) * 100
        if swing_pct < MIN_SWING_PCT or (op - mid_low) < MIN_SWING_ATR * atr:
            continue

        nh = float(df.iloc[-1].high)
        nr = float(df.iloc[-1].rsi)
        if np.isnan(nr):
            continue

        prior_high = float(df.iloc[-4:-1].high.max())
        if nh < prior_high * (1 + MIN_NEW_EXTREME_PCT / 100):
            continue

        price_diff = (nh / op - 1) * 100
        rsi_diff = nr - orsi

        if (price_diff >= MIN_PRICE_PCT and
            rsi_diff <= -MIN_RSI_DIFF and
            nr >= BEAR_SECOND_RSI_MIN):

            return {
                'p': p,
                'time': str(df.iloc[p].open_time),
                'op': op,
                'np': nh,
                'orsi': orsi,
                'nr': nr,
                'pct': price_diff,
                'rd': rsi_diff,
                'bars': bars,
                'swing_pct': swing_pct
            }
    return None

# ====================== TELEGRAM MESAJ ======================
def msg(sym, s, bull):
    typ = '🟢 POZİTİF UYUMSUZLUK — CANLI 2. DİP' if bull else '🔴 NEGATİF UYUMSUZLUK — CANLI 2. TEPE'
    label = 'Dipler' if bull else 'Tepeler'
    priceword = 'daha düşük' if bull else 'daha yüksek'
    rsword = 'daha yüksek' if bull else 'daha düşük'

    return (
        f"{typ}\n\n"
        f"Coin: {sym}\n"
        f"Platform: Binance Futures\n"
        f"Grafik: {TIMEFRAME}\n\n"
        f"1. {'dip' if bull else 'tepe'}: {s['op']:.8g}\n"
        f"2. {'dip' if bull else 'tepe'} (CANLI): {s['np']:.8g}\n"
        f"Fiyat farkı: {s['pct']:+.2f}%\n\n"
        f"1. {'dip' if bull else 'tepe'} RSI: {s['orsi']:.2f}\n"
        f"2. {'dip' if bull else 'tepe'} RSI: {s['nr']:.2f}\n"
        f"RSI farkı: {s['rd']:+.2f}\n\n"
        f"{label} arası mum: {s['bars']}\n"
        f"✓ Fiyat {priceword} dip/tepe yaptı\n"
        f"✓ RSI aynı anda {rsword} kaldı\n"
        f"✓ Arada belirgin swing oluştu (≥ {MIN_SWING_PCT}% ve ≥ {MIN_SWING_ATR} ATR)\n\n"
        f"⚡ 2. {'DİP' if bull else 'TEPE'} ANINDA YAKALANDI\n\n"
        f"⚠️ Mum kapanmadan uyumsuzluk bozulabilir."
    )

async def tg(session, text):
    if not TOKEN or not CHAT:
        return
    try:
        async with session.post(
            f'https://api.telegram.org/bot{TOKEN}/sendMessage',
            json={'chat_id': CHAT, 'text': text},
            timeout=20
        ) as r:
            print(f'Telegram: {r.status}', flush=True)
    except Exception as e:
        print(f'Telegram hata: {e}', flush=True)

# ====================== GEÇMİŞ VERİ ======================
async def history_one(session, sym, sem):
    rows = []
    async with sem:
        for d in range(HISTORY_DAYS, 0, -1):
            date = (datetime.now(timezone.utc) - timedelta(days=d)).strftime('%Y-%m-%d')
            url = f'https://data.binance.vision/data/futures/um/daily/klines/{sym}/{TIMEFRAME}/{sym}-{TIMEFRAME}-{date}.zip'
            try:
                async with session.get(url, timeout=30) as r:
                    if r.status != 200:
                        continue
                    b = await r.read()
                with zipfile.ZipFile(io.BytesIO(b)) as z:
                    with z.open(z.namelist()[0]) as f:
                        raw = f.read().decode()
                frame = pd.read_csv(
                    io.StringIO(raw),
                    header=0 if raw.splitlines()[0].lower().startswith('open_time') else None
                )
                rows += frame.values.tolist()
            except Exception:
                pass

    if not rows:
        return sym, pd.DataFrame()

    # Unique + sort
    rows = sorted({int(x[0]): x for x in rows}.values(), key=lambda x: x[0])
    df = pd.DataFrame(rows).iloc[:, :12]
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'q', 'trades', 'tb', 'tq', 'ig']
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df = add_indicators(df).iloc[-MAX_CANDLES:].reset_index(drop=True)
    return sym, df

async def load():
    print('=' * 60, flush=True)
    print('BINANCE DIVERGENCE SCANNER v2', flush=True)
    print(f'{TIMEFRAME} | min {MIN_BARS} mum | fiyat ≥{MIN_PRICE_PCT}% | swing ≥{MIN_SWING_PCT}% + {MIN_SWING_ATR} ATR', flush=True)
    print('=' * 60, flush=True)

    sem = asyncio.Semaphore(10)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        res = await asyncio.gather(*[history_one(s, x, sem) for x in SYMBOLS])

    for sym, df in res:
        if len(df) >= 50:
            histories[sym] = df
            print(f'OK {sym} → {len(df)} mum', flush=True)

    print(f'\nKullanılabilir coin: {len(histories)}', flush=True)

# ====================== CANLI İŞLEME ======================
async def process(session, k):
    sym = k['s']
    if sym not in histories:
        return

    df = histories[sym]
    t = pd.to_datetime(k['t'], unit='ms', utc=True)

    row = {
        'open_time': t,
        'open': float(k['o']),
        'high': float(k['h']),
        'low': float(k['l']),
        'close': float(k['c']),
        'volume': float(k['v'])
    }

    # Aynı mum güncelleniyorsa güncelle, yeni mumsa ekle
    if len(df) and df.iloc[-1].open_time == t:
        df = df.copy()
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df.loc[df.index[-1], c] = row[c]
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = add_indicators(df.iloc[-MAX_CANDLES:].reset_index(drop=True))
    histories[sym] = df

    # Mum kapandıysa live extreme'leri sıfırla
    if k['x']:
        live_low.pop(sym, None)
        live_high.pop(sym, None)
        return

    low = float(k['l'])
    high = float(k['h'])
    new_low = (sym not in live_low) or (low < live_low[sym])
    new_high = (sym not in live_high) or (high > live_high[sym])

    live_low[sym] = low
    live_high[sym] = high

    now = time.time()

    # Pozitif uyumsuzluk (canlı 2. dip)
    if new_low:
        s = bull_setup(df)
        if s and last_bull.get(sym) != s['time'] and now - last_alert.get((sym, 'b'), 0) >= COOLDOWN:
            last_bull[sym] = s['time']
            last_alert[(sym, 'b')] = now
            await tg(session, msg(sym, s, True))

    # Negatif uyumsuzluk (canlı 2. tepe)
    if new_high:
        s = bear_setup(df)
        if s and last_bear.get(sym) != s['time'] and now - last_alert.get((sym, 's'), 0) >= COOLDOWN:
            last_bear[sym] = s['time']
            last_alert[(sym, 's')] = now
            await tg(session, msg(sym, s, False))

# ====================== WEBSOCKET ======================
async def run_ws():
    syms = list(histories)
    streams = '/'.join(f'{x.lower()}@kline_{TIMEFRAME}' for x in syms)
    url = f'wss://fstream.binance.com/market/stream?streams={streams}'

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        while True:
            try:
                async with s.ws_connect(url, heartbeat=180, receive_timeout=300, max_msg_size=8*1024*1024) as ws:
                    print('BINANCE CANLI BAĞLANDI', flush=True)
                    print(f'Stream sayısı: {len(syms)}', flush=True)
                    print(f'{TIMEFRAME} canlı mumlar takip ediliyor...', flush=True)

                    async for m in ws:
                        if m.type == aiohttp.WSMsgType.TEXT:
                            try:
                                d = json.loads(m.data).get('data', {})
                                k = d.get('k')
                                if d.get('e') == 'kline' and k:
                                    await process(s, k)
                            except Exception as e:
                                print(f'İşleme hatası: {e}', flush=True)
            except Exception as e:
                print(f'WebSocket koptu: {e}', flush=True)
                await asyncio.sleep(5)

# ====================== MAIN ======================
async def main():
    await load()
    print('\nGEÇMİŞ VERİ HAZIR', flush=True)
    print(f'Tarama yapılacak coin: {len(histories)}', flush=True)
    await run_ws()

if __name__ == '__main__':
    asyncio.run(main())
