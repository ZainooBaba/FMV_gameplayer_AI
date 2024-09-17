import pyautogui
from pynput import keyboard

x1 = -1
y1 = -1
def print_cursor_location():
    x, y = pyautogui.position()
    print(f"Cursor Location: X={x}, Y={y}")


def on_press(key):
    global x1, y1
    try:
        if key.char == 'g':
            if x1 == -1 and y1 == -1:
                x1, y1 = pyautogui.position()
                print(f"Cursor Location: X={x1}, Y={y1}")
            else:
                x2, y2 = pyautogui.position()
                print(f"({x1 *2}, {y1*2}, {(x1-x2) * 2}, {(y1-y2) * 2})")
                x1, y1 = -1, -1
    except AttributeError:
        pass


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


def main():
    print("Press 'g' to get the cursor location. Press 'Esc' to exit.")

    # Start listening to keyboard events
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
