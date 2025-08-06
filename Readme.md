# Fountain mvp

Streamlit app that allows users to input image and 

# Running

1. Make sure you have `docker` and `docker-compose` installed. 
2. Run `docker-compose up` in the base directory
3. Visit [localhost:8501](http://localhost:8501)

# Project Structure 
## `data/`
- Data files sourced from these data sets
  - [Otlis - Kaggle](https://www.kaggle.com/datasets/erdalbasaran/eardrum-dataset-otitis-media)
  - [Otoscopedata - Kaggle](https://www.kaggle.com/datasets/omduggineni/otoscopedata)
  - [Roboflow](https://universe.roboflow.com/otoscope/digital-otoscope)
  - [Figshar](https://figshare.com/articles/dataset/eardrum_zip/13648166/1?file=26200970)
  - [Sumotosima](https://github.com/anas2908/Sumotosima)



# Classes
- [ ] Need a list of all of these and training data from bio expert
- Normal Tympanic Membrane (HE/Normal)
    - UCI, VanAk, Ebasaran, Sumotosima
- Acute Otitis Media (AOM)
    - UCI, VanAk, Ebasaran, Sumotosima
- Myringosclerosis (MG/MK/Tympanskleros)
    - UCI, Sumotosima, VanAk, Ebasaran
- Chronic Otitis Media (COM/CSOM)
    - UCI, Sumotosima, VanAk, Ebasaran
- Cerumen Impaction/Earwax (CI/EW)
    - UCI, Sumotosima, VanAk, Ebasaran
- Tympanostomy Tubes/Ear Ventilation Tube (TT/EVT)
    - VanAk, Ebasaran
- Otitis Externa (OE)
    - VanAk, Ebasaran
- Foreign Object/Foreign Bodies
    - Ebasaran
- Pseudo Membranes
    - Ebasaran

Missing from this as no data
- Bullous Myringitis - Characterized by fluid-filled blisters on the tympanic membrane, often associated with viral or mycoplasma infections
- Tympanic Membrane Perforation - A hole or rupture in the eardrum that can be clearly visible on otoscopic examination
- Tympanic Membrane Retraction - Where the eardrum appears pulled inward due to negative pressure in the middle ear
- Cholesteatoma - A skin growth that occurs in the middle ear behind the eardrum, appearing as a white/yellow mass
- Granulation Tissue - Red, inflamed tissue that can form during healing processes
- Hemorrhagic Bullae - Blood-filled blisters on the tympanic membrane often seen in influenza or severe infections
- Glomus Tumors - Reddish vascular masses visible behind the tympanic membrane
- Exostoses/Osteomas - Bony growths in the ear canal that narrow the passage
- Fungal Infections (Otomycosis) - Typically showing white/black/green fungal elements in the ear canal
- Barotrauma - Changes to the eardrum from pressure injuries, often appearing as hemorrhage or retraction
- Serous Otitis Media (Middle Ear Effusion) - Fluid buildup behind the eardrum without acute inflammation
- Attic Retraction Pockets - Localized retractions of the tympanic membrane, often in the pars flaccida
- Ossicular Chain Disruption - Sometimes visible through a translucent or perforated tympanic membrane


# Notes 
With the different ear ifnections and the such they all present with
1. Detectable Symptoms (from image)
2. Undetectable Symptoms such as fever, pain or ear lose
