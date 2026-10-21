import argparse
import threading
import webbrowser

from pal.debug.quiet import quiet_startup

quiet_startup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Open the debug preview (camera + live log) in the browser")
    parser.add_argument("--preview-port", type=int, default=4200)
    parser.add_argument("--text", action="store_true",
                        help="Read from stdin instead of mic (no STT)")
    parser.add_argument("--knowledge", action="append", default=[], metavar="PATH",
                        help="Extra reference material (.md/.txt file or folder) for this run, on top of data/knowledge/")
    parser.add_argument("--yolo", action="store_true",
                        help="Run object detection and send a crop of the object nearest the center, instead of the whole frame")
    parser.add_argument("--remote", action="store_true",
                        help="Use a remote device (the phone app, over LiveKit) as mic, camera and speaker instead of this laptop")
    parser.add_argument("--verbose", action="store_true",
                        help="Show debug lines (timings) in the terminal too; the log file always has them")
    args = parser.parse_args()

    from pal.log import setup
    setup(verbose=args.verbose)

    # Imports after argparse so any module-level setup can react to flags.
    from pal.agent import Pal

    if args.preview:
        from pal.debug.preview import preview
        preview.start(port=args.preview_port)
        webbrowser.open(f"http://127.0.0.1:{args.preview_port}")

    pal = Pal(knowledge=args.knowledge, remote=args.remote)
    pal.run(text_mode=args.text, detect=args.yolo)
    try:
        threading.Event().wait()  # keep main thread alive
    except KeyboardInterrupt:
        print()  # end the ^C line; models are released on exit

if __name__ == "__main__":
    main()