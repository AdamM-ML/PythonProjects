from random import randint
from tkinter import *
from tkinter import Tk
import tkinter.scrolledtext as scrolledtext
import logging
import datetime


"""Instance of tkinter-based UI - class definition"""

class TkUi:
    root = Tk()
    root.title("Ui")
    txt = scrolledtext.ScrolledText(root, height=8, undo=True, font=("Verdana", "16"),
                                    relief='solid', highlightthickness=1,
                                    highlightbackground="Black")
    txt.configure(padx=190, bg="white", spacing1=2, spacing2=3, spacing3=2, wrap=WORD, state="disabled")
    txt.pack(expand=True, fill='both', pady=12)
    root.configure(background="LightSkyBlue2")


    @classmethod
    def centerWindow(cls, w=300, h=200):
        ws = cls.root.winfo_screenwidth()
        hs = cls.root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2) - 35
        cls.root.geometry('%dx%d+%d+%d' % (w, h, x, y))

    #root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file='images/iconchat.ico'))


"""Class containing mathematic operations/functions based on randomly generated collection of numbers"""
class MathOps:

    def __init__(self, numbers):
        self.numbers = numbers

    @staticmethod
    def numbers_generator(quantity):
        while quantity > 0:
            yield randint(1, 10)
            quantity = quantity-1

    @staticmethod
    def quadratic_function_constructor(numbers=(1, 1, 1)):
        return lambda x: numbers[0] * x ** 2 + numbers[1] * x + numbers[2]


"""Putting MathOps operations results into the UI and logging them into file"""
def display_handler(math_instance, ui, f_name, collection, x):
    output = math_instance.quadratic_function_constructor(collection)
    ui.txt.configure(state="normal")
    ui.txt.insert(INSERT, f'\nResult of the function {f_name}\nfor the collection: {collection}\ngives the '
            f'result: {output(x)}\n\n')
    ui.txt.configure(state="disabled")

    """Logging. Using 'n' and 'n_modified' to create log files with next collection of results
     each time the program is executed"""
    LOGGING_FORMAT="%(levelname)s %(asctime)s - %(message)s"
    n = datetime.datetime.now()
    n_modified = str(n.strftime("%H_%M_%S"))
    logging.basicConfig(filename=f"log{n_modified}",
                        level=logging.DEBUG,
                        format=LOGGING_FORMAT,
                        filemode="w")
    logger = logging.getLogger()
    logging.info(ui.txt.get("1.0", END))


"""Starting UI"""
ui = TkUi()
ui.centerWindow(ui.root.winfo_screenwidth() - 280, ui.root.winfo_screenheight() - 280)

"""Preparing collection of numbers"""
nums = MathOps.numbers_generator(12000)
numbers_collection1 = (next(nums), next(nums), next(nums))

math1 = MathOps(numbers_collection1)
display_handler(math1, ui, MathOps.quadratic_function_constructor.__qualname__, numbers_collection1, 5)

numbers_collection2 = tuple(map(lambda x: x * 2, numbers_collection1))

math2 = MathOps(numbers_collection2)
display_handler(math2, ui, MathOps.quadratic_function_constructor.__qualname__, numbers_collection2, 3)

numbers_collection3 = tuple(map(lambda x: x * 3, numbers_collection2))

math3 = MathOps(numbers_collection3)
display_handler(math2, ui, MathOps.quadratic_function_constructor.__qualname__, numbers_collection3, 3)

numbers_collection4 = tuple(map(lambda x: x * 4, numbers_collection3))

math4 = MathOps(numbers_collection4)
display_handler(math2, ui, MathOps.quadratic_function_constructor.__qualname__, numbers_collection4, 3)

ui.root.mainloop()

