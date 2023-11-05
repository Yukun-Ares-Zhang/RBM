import xlwt
f = xlwt.Workbook()
work_sheet = f.add_sheet("Test")
work_sheet.write(0, 0, "Hello World")
test_list = [str(i) for i in range(5)]
work_sheet.write_rich_text(1, 0, test_list)
work_sheet.merge(2, 3, 0, 3)
work_sheet.write_merge(4, 4, 0, 3, "Merged cell data")
f.save("Test.xls")