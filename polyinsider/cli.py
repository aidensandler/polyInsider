"""CLI entry point: python -m polyinsider.cli --help"""
import argparse
import json
import threading
import time
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from functools import partial
from .core import write_training
from .prepare import prepare_snapshots,prepare_snapshots_streaming
from .demo import build_demo
from .live import capture_live,evaluate_saved_predictions,verify_audit_chain
from .wallet_model import train_wallet_file
from .history_download import download_v2_fills

ROOT=Path(__file__).resolve().parents[1]
def main(argv=None):
    parser=argparse.ArgumentParser(description='PolyInsiders pre-resolution tracker, research prototype')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('demo',help='Build clearly labeled synthetic training data and demo dashboard')
    dl=sub.add_parser('download-trades',help='Download V2 taker fills for supplied market ids; needs a working Internet connection')
    dl.add_argument('--markets',required=True);dl.add_argument('--out',required=True)
    dl.add_argument('--max-pages',type=int,default=300)
    prep=sub.add_parser('prepare',help='Convert original v2 trade CSV and independently verified market metadata to frozen snapshots')
    prep.add_argument('--trades',required=True);prep.add_argument('--markets',required=True)
    prep.add_argument('--out',default=str(ROOT/'data'/'prepared_history.csv'))
    prep.add_argument('--lookback-hours',type=int,default=48)
    prep.add_argument('--stream',action='store_true',help='Use bounded-memory CSV chunks + disk-based aggregation for millions of fills')
    t=sub.add_parser('train',help='Train on historical pre-resolution snapshots with resolved outcomes')
    t.add_argument('--csv',required=True);t.add_argument('--source',default=None)
    t.add_argument('--cutoff',default=None);t.add_argument('--out',default=str(ROOT/'data'/'run'))
    wt=sub.add_parser('train-wallets',help='Fit a second, chronologically evaluated model of subsequent wallet outperformance')
    wt.add_argument('--csv',required=True);wt.add_argument('--source',default=None);wt.add_argument('--asof',default=None)
    wt.add_argument('--min-prior',type=int,default=3);wt.add_argument('--min-future',type=int,default=3)
    wt.add_argument('--out',default=str(ROOT/'data'/'run'))
    check=sub.add_parser('verify-log',help='Verify SHA-256 chain of captured forecast snapshots')
    check.add_argument('--path',default=str(ROOT/'data'/'run'/'predictions_audit.jsonl'))
    live=sub.add_parser('snapshot',help='Poll open markets and save timestamped forecasts')
    live.add_argument('--limit',type=int,default=8);live.add_argument('--model',default=str(ROOT/'data'/'run'/'model.json'))
    live.add_argument('--allow-demo-model',action='store_true')
    serve=sub.add_parser('serve',help='Run local website; optional periodic live capture')
    serve.add_argument('--port',type=int,default=8000)
    serve.add_argument('--live',action='store_true',help='Refresh live open-market forecasts while the local server is running')
    serve.add_argument('--interval',type=int,default=120);serve.add_argument('--limit',type=int,default=8)
    serve.add_argument('--allow-demo-model',action='store_true')
    ev=sub.add_parser('evaluate',help='Compare saved pre-resolution predictions to independently supplied resolution labels')
    ev.add_argument('--predictions',required=True);ev.add_argument('--resolutions',required=True)
    ev.add_argument('--out',default=str(ROOT/'data'/'run'/'live_evaluation.json'))
    args=parser.parse_args(argv)
    if args.command=='demo':
        print(json.dumps(build_demo(ROOT),indent=2));return
    if args.command=='download-trades':
        print(json.dumps(download_v2_fills(args.markets,args.out,args.max_pages),indent=2));return
    if args.command=='prepare':
        procedure=prepare_snapshots_streaming if args.stream else prepare_snapshots
        print(json.dumps(procedure(args.trades,args.markets,args.out,args.lookback_hours),indent=2));return
    if args.command=='train':
        print(json.dumps(write_training(args.csv,args.out,args.source,args.cutoff),indent=2));return
    if args.command=='train-wallets':
        print(json.dumps(train_wallet_file(args.csv,args.out,args.source,args.asof,args.min_prior,args.min_future),indent=2));return
    if args.command=='verify-log':
        print(json.dumps(verify_audit_chain(args.path),indent=2));return
    if args.command=='snapshot':
        print(json.dumps(capture_live(args.model,ROOT/'web',args.limit,allow_demo_model=args.allow_demo_model),indent=2));return
    if args.command=='evaluate':
        print(json.dumps(evaluate_saved_predictions(args.predictions,args.resolutions,args.out),indent=2));return
    if args.command=='serve':
        if args.live:
            if args.interval<30:parser.error('--interval must be at least 30 seconds')
            def polling():
                while True:
                    try:
                        result=capture_live(ROOT/'data'/'run'/'model.json',ROOT/'web',
                              args.limit,allow_demo_model=args.allow_demo_model)
                        print(f"Snapshot: {len(result['markets'])} markets @ {result['updated_at']}")
                    except Exception as exc:print(f'Capture failed (retaining last snapshot): {exc}')
                    time.sleep(args.interval)
            threading.Thread(target=polling,daemon=True).start()
        h=partial(SimpleHTTPRequestHandler,directory=str(ROOT/'web'))
        server=ThreadingHTTPServer(('127.0.0.1',args.port),h)
        print(f'Open http://localhost:{args.port}/ — Ctrl+C to stop')
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()
if __name__=='__main__': main()
