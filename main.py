import argparse
import threading

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Enable debug preview at http://localhost:4200")
    parser.add_argument("--preview-port", type=int, default=4200)
    parser.add_argument("--text", action="store_true",
                        help="Read from stdin instead of mic (no STT)")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Skip object detection; send the whole frame to the vision model")
    args = parser.parse_args()

    # Imports after argparse so any module-level setup can react to flags.
    from startup import filtered_stderr

    with filtered_stderr():  # cv2/av duplicate-dylib warnings land here
        from agents.scout import Scout

        if args.preview:
            from pal.debug.preview import preview
            preview.start(port=args.preview_port)

    scout = Scout()
    scout.run(text_mode=args.text, detect=not args.no_yolo)
    threading.Event().wait()  # keep main thread alive

if __name__ == "__main__":
    main()