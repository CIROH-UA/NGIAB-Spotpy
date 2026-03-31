from cli import main as cli_main


def main() -> int:
    return int(cli_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
