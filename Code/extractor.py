import PyPDF2
import re
def extract_from_pdf(pdf_path):
    '''Extracts Data from the PDF file'''
    file=open(pdf_path,'rb')
    pdf_reader=PyPDF2.PdfReader(file)
    content=''
    for i in pdf_reader.pages:
        content+=i.extract_text()
    file.close()
    return content


content=extract_from_pdf('/home/mav_27/Internship Assignment/Data/data/data/BANKING/3547447.pdf')
paragraphs=content.split('\n\n')
job_role=paragraphs[0].split('\n')[0]                                                       # Extracting the Job Role
skills=[]
education=''
lines=re.split(r'[\n,;]',paragraphs[0])
lines2=re.split(r'[\n\n,\n,:]',paragraphs[0])
for i,line in enumerate(lines):                                                             # Extraction of Skills from the extracted data
    if 'Skills' in line:
        skills.append(lines[max(i+1,0):i+70])
for i,line in enumerate(lines2):                                                            # Extraction of Education from the extracted data
    if 'Education' in line:
        education=(lines2[i+1:i+9])
print(job_role)
print(skills)
print(education)
with open('extracted_data.txt','+w') as file:                                               # Writing all the extracted data to a text file for further use
    file.writelines('Job_role : '+job_role)
    file.writelines('\nSkills : '+str(skills))
    file.writelines("\nEducation : "+str(education))
