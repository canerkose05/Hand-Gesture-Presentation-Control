import requests


SERVER_URL = "http://127.0.0.1:8000/command/"

SUPPORTED_COMMANDS = [
    "swipe_left",
    "swipe_right",
    "rotate_right",
]


def send_command(action: str) -> bool:
    if action not in SUPPORTED_COMMANDS:
        print(f"Unsupported command: {action}")
        return False

    url = SERVER_URL + action

    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            print(f"Command '{action}' sent successfully.")
            return True

        print(f"Server returned status {response.status_code} for '{action}'")
        return False

    except requests.RequestException as exc:
        print(f"Failed to send command '{action}': {exc}")
        return False


def main():
    print("Available commands:")
    for i, command in enumerate(SUPPORTED_COMMANDS, start=1):
        print(f"{i}. {command}")
    print("0. Exit")

    while True:
        choice = input("Enter command number: ").strip()

        if choice == "0":
            print("Exiting...")
            break

        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(SUPPORTED_COMMANDS):
                send_command(SUPPORTED_COMMANDS[index])
                continue

        print("Invalid command number. Please try again.")


if __name__ == "__main__":
    main()