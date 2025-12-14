from argostranslate import translate

def translate_text(text, src_lang="en", tgt_lang="hi"):
    try:
        return translate.translate(text, src_lang, tgt_lang)
    except:
        return text
