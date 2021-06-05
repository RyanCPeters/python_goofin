import PySimpleGUI as sg

def build_gui_window():
    sg.theme('Topanga')      # Add some color to the window

    # Very basic window.  Return values using auto numbered keys

    layout = [
        [sg.Text('Func id', size=(15, 1)),  sg.InputText(key="choice")],
        [sg.Text('color idx', size=(15, 1)),  sg.InputText(key="c_pos")],
        [sg.Text('zoom', size=(15, 1)),  sg.InputText(key="zoom")],
        [sg.Text('R', size=(15, 1)),  sg.InputText(key="R")],
        [sg.Text('r', size=(15, 1)),  sg.InputText(key="r")],
        [sg.Text('d', size=(15, 1)),  sg.InputText(key="d")],
        [sg.Text('a', size=(15, 1)),  sg.InputText(key="a")],
        [sg.Text('b', size=(15, 1)),  sg.InputText(key="b")],
        [sg.Text('c', size=(15, 1)),  sg.InputText(key="c")],
        [sg.Text('e', size=(15, 1)),  sg.InputText(key="e")],
        [sg.Text('j', size=(15, 1)),  sg.InputText(key="j")],
        [sg.Text('k', size=(15, 1)),  sg.InputText(key="k")],
        [sg.Text('j2', size=(15, 1)), sg.InputText(key="j2")],
        [sg.Text('i2', size=(15, 1)), sg.InputText(key="i2")],
        [sg.Submit(), sg.Cancel()]
    ]

    window = sg.Window('Parameter specification window', layout)
    return window
    # try:
    #     event = -1
    #     while event is not None:
    #         event, values = window.read()
    #
    # finally:
    #     window.close()
    # print(event, values[0], values[1], values[2])    # the input data looks like a simple list when auto numbered