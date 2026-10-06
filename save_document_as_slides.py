import pandas as pd
from pptx import Presentation
from pptx.util import Inches
from docx import Document
from os import path
prs = Presentation()


summary = pd.read_excel(rf"C:\Users\rlong5\Desktop\FHS_JHS_manuscript\FHSJHS_PCAaxes_LR_CV_allData_3PCs_summary.xlsx")
# metrics #"FULLDATA_LR_metrics_by_featureset_3PCs.csv")
top15 = pd.read_excel(rf"C:\Users\rlong5\Desktop\FHS_JHS_manuscript\Top15_radiomics_driving_PCA_FULLDATA_CHD_Advance_Combine_Features_3PCs.xlsx")

PATH_Q = rf"C:\Users\rlong5\Desktop\FHS_JHS_manuscript"



print("\nSUMMARY")
print(summary.head(15).to_string())


print("\nTOP15 FEATURES")
print(top15.head(15).to_string())

# Create Word document
doc = Document()
doc.add_heading('Top 15 Rows', level=1)

top15 = top15.head(15)

# Create table
table = doc.add_table(rows=1, cols=len(top15.columns))
table.style = 'Table Grid'

# Header row
for i, col in enumerate(top15.columns):
    table.rows[0].cells[i].text = str(col)

# Data rows
for _, row in top15.iterrows():
    cells = table.add_row().cells
    for i, value in enumerate(row):
        cells[i].text = str(value)

# Save Word document
doc.save(path.join(PATH_Q, "top15_rows.docx"))

print("Saved: top15_rows.docx")

# Header row
for i, col in enumerate(summary.columns):
    table.rows[0].cells[i].text = str(col)

# Data rows
for _, row in top15.iterrows():
    cells = table.add_row().cells
    for i, value in enumerate(row):
        cells[i].text = str(value)

# Save Word document
doc.save(path.JOIN(PATH_Q,"summary_rows.docx"))

print("Saved: summary_rows.docx")
print(PATH_Q)