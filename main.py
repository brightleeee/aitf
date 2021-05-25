import xlwings as xw
import pandas as pd



def main1():
    wb = xw.Book.caller()
    sheet = wb.sheets[0]

    if sheet["A1"].value == "Hello xlwings!":
        sheet["A1"].value = "Bye xlwings!"
    else:
        sheet["A1"].value = "Hello xlwings!"

    print(sheet["A1"].value)
    sheet2 = sheet.range('A1').options(pd.DataFrame, index=False, header=False, expand='table').value
    print(sheet2)

    sheet3 = sheet2.to_html()
    print(sheet3)
    return sheet3

if __name__ == "__main__":
    xw.Book("pyxl.xlsm").set_mock_caller()
    main1()