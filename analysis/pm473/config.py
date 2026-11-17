"""Endpoints, contract addresses, and project-wide constants.

Verified against Polymarket / Etherscan docs as of late October 2026.
Polymarket changes these often -- re-check docs.polymarket.com before a big pull.
"""
from datetime import datetime, timezone

# --- Polymarket HTTP services (all public for reads, no key needed) ---------
GAMMA_URL = "https://gamma-api.polymarket.com"   # market/event metadata
DATA_URL = "https://data-api.polymarket.com"     # trades, activity, price history (use /v2 routes)
CLOB_URL = "https://clob.polymarket.com"         # live order book only (no historical books)

# Data API v1 was retired 2026-10-24. CLOB /prices-history shuts off 2026-10-30.
# Everything here uses Data API v2: {data, pagination} envelope, snake_case,
# cursor pagination (no offset), lowercase "true"/"false" booleans.

# --- On-chain (Polygon PoS, chain id 137) -----------------------------------
POLYGON_CHAIN_ID = 137
# PolygonScan's own API was switched off in Aug 2025; use Etherscan API V2 with chainid=137.
ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"

# Exchange contracts that emit OrderFilled. CLOB V2 cut over 2026-04-28 ~11:00 UTC
# with new contracts and a NEW OrderFilled event layout.
EXCHANGES = {
    "ctf_v1":     "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",
    "negrisk_v1": "0xc5d563a36ae78145c45a50134d48a1215220f80a",
    "ctf_v2":     "0xe111180000d2663c0091e4f400237545b87b996b",
    "negrisk_v2": "0xe2222d279d744050d28e00520010520000310f59",
}
CONDITIONAL_TOKENS = "0x4d97dcd97ec945f40cf65f87097ace5ea0476045"
NEG_RISK_ADAPTER = "0xd91e80cf2e7be2e162c6513ced06f1dd0da35296"

# Addresses that can show up as a "wallet" but are infrastructure, not traders.
NON_TRADER_ADDRESSES = set(EXCHANGES.values()) | {CONDITIONAL_TOKENS, NEG_RISK_ADAPTER}

CLOB_V2_CUTOVER = datetime(2026, 4, 28, 11, 0, tzinfo=timezone.utc)

# Protocol V2 markets (PositionManager / ExchangeV3) start being created ~2026-11-02.
# Gamma tags each market with `version`; this project restricts to "v1" (CTF) markets.
ALLOWED_PROTOCOL_VERSIONS = {"v1", None}

# Collateral (pUSD / USDC.e) and outcome tokens both use 6 decimals.
TOKEN_DECIMALS = 6

# --- Politeness / rate limiting --------------------------------------------
# Published limits (per 10 s, per IP): Data API ~800 overall, /v2/trades ~300.
# Stay well under: ~4 requests/second.
MIN_SECONDS_BETWEEN_REQUESTS = 0.25

# --- Scheduled information releases ---------------------------------------
# FOMC policy statements, 2:00 pm ET on day 2 (federalreserve.gov, 2026).
# Jan 28 is EST (UTC-5); the rest are EDT (UTC-4).
FOMC_STATEMENTS_UTC = [
    "2026-01-28T19:00:00Z", "2026-03-18T18:00:00Z", "2026-04-29T18:00:00Z",
    "2026-06-17T18:00:00Z", "2026-07-29T18:00:00Z", "2026-09-16T18:00:00Z",
    "2026-10-28T18:00:00Z", "2026-12-09T19:00:00Z",
]
