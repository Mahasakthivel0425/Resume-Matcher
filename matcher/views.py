import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

def matchresume(request):
    if request.method == 'POST':
        job_description = request.POST['job_description']
        resume_files = request.FILES.getlist('resumes')

        if not resume_files or not job_description:
            return render(request, 'matchresume.html', {'message': "Please upload resumes and enter a job description."})

        resumes = []
        for resume_file in resume_files:
            fs = FileSystemStorage(location=settings.MEDIA_ROOT + '/uploads/')
            filename = fs.save(resume_file.name, resume_file)
            file_path = os.path.join(fs.location, filename)
            resumes.append(extract_text(file_path))

        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # Get top 5 resumes and their similarity scores
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [(resume_files[i].name, round(similarities[i], 2)) for i in top_indices]

        return render(request, 'matchresume.html', {'message': "Top matching resumes:", 'top_resumes': top_resumes})

    return render(request, 'matchresume.html')
