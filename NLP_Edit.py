import spacy
from spacy.tokens import Span
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import EditFunctions # Assurez-vous que EditFunctions contient vos fonctions d'édition d'image
from nltk.corpus import wordnet as wn
import nltk
import re
import matplotlib.colors as mcolors
from matplotlib.colors import CSS4_COLORS
from spellchecker import SpellChecker



try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading the wordnet data...")
    nltk.download('wordnet')

#Charger le modèle spaCy pour les dépendances syntaxiques
#nlp = spacy.load("en_core_web_trf")


try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    from spacy.cli import download
    download("en_core_web_trf")
    nlp = spacy.load("en_core_web_trf")


# Charger le modèle spaCy
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     from spacy.cli import download
#     download("en_core_web_sm")
#     nlp = spacy.load("en_core_web_sm")
# Liste des couleurs prédéfinies (CSS4)
colors = list(CSS4_COLORS.keys())


# Initialiser le correcteur orthographique
spell = SpellChecker()

# Fonction pour corriger les fautes d'orthographe dans une instruction
def correct_spelling(instruction, special_terms):
    print("instruction: ", instruction)
    words = instruction.split()
    corrected_words = []

    for word in words:
        # Ne pas corriger les termes spéciaux
        if word in special_terms:
            corrected_words.append(word)
        else:
            corrected_word = spell.correction(word)
            if corrected_word is not None:
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)  # Use the original word if correction is None
    
    corrected_instruction = " ".join(corrected_words)
    return corrected_instruction



class ColourExtractorStrict:
    """Extract colours along with adjectives"""

    def __init__(self, colours):
        self.colours = colours
        self.pos_ok = ['ADJ', 'NOUN']
        self.tagger = spacy.load('en_core_web_sm')

    def get(self, string):
        extracted = set()
        doc = self.tagger(string.lower())
        pairs = [(word.text, word.pos_) for word in doc]
        for index, pair in enumerate(pairs):
            text, pos = pair
            if text in self.colours:
                text_ahead = self.look_ahead(pairs=pairs, index=index)
                text_behind = self.look_behind(pairs=pairs, index=index,
                                               colour_pos=pos)
                if text_behind:
                    text_behind.append(text)
                    if text_ahead:
                        text_behind.extend(text_ahead)
                        extracted.add(' '.join(text_behind))
                    else:
                        extracted.add(' '.join(text_behind))
                elif text_ahead:
                    extracted.add(' '.join([text] + text_ahead))
                else:
                    extracted.add(text)

        return extracted if extracted else False

    def look_ahead(self, pairs, index):
        ahead = list()
        for text, pos in pairs[index + 1:]:
            if pos in self.pos_ok:
                ahead.append(text)
            else:
                break

        return ahead if ahead else False

    def look_behind(self, pairs, index, colour_pos):
        behind = list()
        for text, pos in reversed(pairs[:index]):
            if pos in self.pos_ok:
                behind.append(text)
            else:
                break

        return list(reversed(behind)) if behind else False


class NLP_Editor:
    def __init__(self):
        self.generate_synonyms = self.get_synonyms('generate').union(self.get_synonyms('create')).union(self.get_synonyms('produce')).union(self.get_synonyms('draw')).union(self.get_synonyms('illustrate')).union(self.get_synonyms('visualize')).union(self.get_synonyms('construct'))
        print(f"Combined generate synonyms: {self.generate_synonyms}")
        self.edit_synonyms = self.get_synonyms('edit').union(self.get_synonyms('modify')).union(self.get_synonyms('change')).union(self.get_synonyms('add')).union(self.get_synonyms('replace')).union(self.get_synonyms('put'))
        self.extract_synonyms=self.get_synonyms('extract')
        self.merge_synonyms=self.get_synonyms('merge').union(self.get_synonyms('join')).union(self.get_synonyms('integrate')).union(self.get_synonyms('fuse')).union(self.get_synonyms('append'))
        print(f"merge synonyms: {self.merge_synonyms}")
        self.colour_extractor = ColourExtractorStrict(colors)
        self.actions=[]
        self.generate_keywords = {'generate', 'create', 'produce'}
        self.edit_keywords = {'rotate', 'flip', 'change', 'extract', 'merge'}


    @staticmethod
    def get_synonyms(word):
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        #print("synonyms: ",synonyms)
        return synonyms

    @staticmethod
    def is_image(token_text):
        return re.match(r'pic\d+', token_text) is not None
    
    @staticmethod
    def is_angle(token_text):
        return re.match(r'\d+', token_text) is not None

    def is_generate_word(self,word):
        return word in self.generate_synonyms

    def is_edit_word(self,word):
        return word in self.edit_synonyms
    
    
    def is_extract_word(self, word):
        return word in self.extract_synonyms

    def is_merge_word(self, word):
        return word in self.merge_synonyms

    def extract_image_ids(self, text):
        pattern = re.compile(r'pic\d+')
        return pattern.findall(text)

    
    
        
    def extract_objects(self,token, action):
        for child in token.children:
            print(f"  Child token: {child.text} (DEP: {child.dep_}, POS: {child.pos_})")
            if self.is_image(child.text):
                print("pic: ",re.compile(r'pic\d+'))
                action["image_id"].append(child.text)
                print(f"  Child token: {child.text} (DEP: {child.dep_}, POS: {child.pos_})")
                self.extract_objects(child, action)
            elif child.dep_ in {"npadvmod", "advmod"}:  
                action["object"].append(child.text)
            elif child.dep_ in { "nummod","amod"}:  
                action["object"].append(child.text+' '+child.head.text)
            elif child.dep_ =="pobj" or (child.dep_== "dobj" and child.pos_!="PRON"):
                action["object"].append(child.text)
                print(f"    Added object: {child.text}")
                self.extract_objects(child, action)
            elif child.dep_ == "prep":
                print("child: ",child)
                self.extract_objects(child, action)
            

    def extract_actions(self,doc):
        actions = []
        for token in doc:
            lemma = token.lemma_.lower()
            print(f"Token text: {token.text}, Lemma: {lemma}, POS: {token.pos_}, DEP: {token.dep_}")
            if (token.pos_ == "VERB") or (lemma in {"rotate", "merge", "color"}):
                token.pos_ = "VERB"
                action = {"action": lemma, "image_id": [],"object": []}
                print(f"Action detected: {action}")
                self.extract_objects(token, action)
                actions.append(action)
        print("actions:", actions)
        return actions


    

    
   


    def process_instruction(self, instruction):
        
        doc = nlp(instruction)
        image_ids = self.extract_image_ids(instruction)
        print("image_ids: ",image_ids)
        action = None
        color = None
        angle = None
        results = []
        print("doc: ",doc)
        extracted_colors = self.colour_extractor.get(instruction)
        if extracted_colors:
            listOfcolor = list(extracted_colors)  # Prendre la première couleur extraite
            print("color: ",listOfcolor)

        actions=self.extract_actions(doc)
        print("actions: ",actions)
        for action in actions:
            print(f"Verbe: {action['action']}")


            verb = action['action']
            objects = action["object"]
            image_id=action["image_id"]

            if verb in ["rotate", "turn"]:
                for obj in objects:
                    if re.findall(r'\d+', obj):
                        angle = int(re.findall(r'\d+', obj)[0])
                results.append({"action": "Rotate", "image_id": image_id, "angle": angle})

            elif verb in ["color", "colour",  "colore"] or (verb =="change" and "color" in objects):
                for obj in objects:
                    if listOfcolor:
                        if obj in listOfcolor:
                            color = obj
                    results.append({"action": "Change Color", "image_id": image_id, "color": color})


            elif self.is_edit_word(verb) and ("background" in objects):
                if extracted_colors:
                    for obj in objects:
                        if listOfcolor:
                            if obj in listOfcolor:
                                color = obj
                    results.append({"action": "Change Background", "image_id": image_id, "color": color})
                else:
                    results.append({"action": "Change Background", "image_id": image_id, "color": color})


            elif verb in ["flip"]:
                if "left" in instruction or "right" in instruction:
                    #if image_id in image_ids:
                        results.append({"action": "flip left-right", "image_id": image_id})
                elif "up" in instruction or "down" in instruction:
                    #if image_id in image_ids:
                        results.append({"action": "flip up-down", "image_id": image_id})

            elif self.is_extract_word(verb):
                for obj in objects:
                    if obj in image_ids:
                        image_id = obj
                    else:
                        results.append({"action": "extract", "image_id": image_id, "object": obj})

            elif self.is_merge_word(verb):
                    results.append({"action": "merge", "image_ids": image_id})


        
    
    

        print(f"Processed actions: {results}")

        return results

    def apply_edit(self, choice, img,action):
        if choice == "Change Color" :
            print("img: ",img)
            color = mcolors.to_hex(action['color'])
            return EditFunctions.change_color(img, color)
        elif choice == "Rotate" :
            if action['angle'] is not None:
                angle=int(action['angle'])
            else:
                angle=action['angle']
            return EditFunctions.rotate_image(img, angle)
        elif choice == "flip left-right":
            print("img: ",img)
            return EditFunctions.flip_image_lr(img)
        elif choice == "flip up-down":
            return EditFunctions.flip_image_ud(img)
        else:
            return img

    def parse_instruction(self, instruction):
        print(f"Instruction received: {instruction}")
        doc = nlp(instruction)
        actions = []
        current_segment = []
        current_type = None  # Can be 'generate' or 'edit'

        for token in doc:
            lemma = token.lemma_.lower()
            print(f"Processing token: '{token.text}', Lemma: '{lemma}', POS: '{token.pos_}', DEP: '{token.dep_}'")
            
            if self.is_generate_word(lemma) and token.pos_ =='VERB' :
                print(f"Found generate keyword: '{lemma}'")
                if current_segment and current_type != 'generate':
                    # Add accumulated edit segment to actions but do not process yet
                    segment_text = ' '.join(current_segment).strip()
                    print(f"Accumulating edit segment: {segment_text}")
                    segment_text='please '+segment_text
                    edit_actions = self.process_instruction(segment_text)
                    actions.extend(edit_actions)

                    current_segment = []
                current_type = 'generate'
            elif lemma in self.edit_keywords:
                print(f"Found edit keyword: '{lemma}'")
                if current_segment and current_type != 'edit':
                    # Add accumulated generate segment to actions
                    segment_text = ' '.join(current_segment).strip()
                    print(f"Adding generate segment to actions: {segment_text}")
                    generate_segments = self.split_generate_actions(segment_text)
                    for segment in generate_segments:
                        actions.append({
                            'action': 'generate',
                            'instruction': segment.strip()
                        })
                    current_segment = []
                current_type = 'edit'

            current_segment.append(token.text)
            print(f"Current segment: {' '.join(current_segment).strip()} (Type: {current_type})")

        # Process the final accumulated segment
        if current_segment:
            action_type = 'generate' if current_type == 'generate' else 'edit'
            segment_text = ' '.join(current_segment).strip()
            print(f"Adding final segment to actions: {segment_text} (Type: {action_type})")
            if action_type == 'generate':
                generate_segments = self.split_generate_actions(segment_text)
                for segment in generate_segments:
                    actions.append({
                        'action': 'generate',
                        'instruction': segment.strip()
                    })
            else:
                # Process all edit actions at once
                segment_text='please '+segment_text
                edit_actions = self.process_instruction(segment_text)
                actions.extend(edit_actions)

        print(f"Final actions list: {actions}")
        return actions


    def split_generate_actions(self, segment_text):
        """ Splits a segment containing multiple generate actions into separate generate instructions. """
        print(f"Splitting generate actions in segment: {segment_text}")
        generate_segments = []
        temp_segment = []
        for word in segment_text.split():
            if word.lower() in self.generate_keywords:
                if temp_segment:
                    generate_segments.append(' '.join(temp_segment))
                    temp_segment = []
            temp_segment.append(word)
        if temp_segment:
            generate_segments.append(' '.join(temp_segment))
        print(f"Generated segments: {generate_segments}")
        return generate_segments
