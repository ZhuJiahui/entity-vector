# -*- coding: utf-8 -*-

import itertools
import logging
import mwparserfromhell
import mwparserfromhell.nodes
import numpy as np
import re
from collections import defaultdict, Counter
from dawg import BytesDAWG
from repoze.lru import lru_cache

logger = logging.getLogger(__name__)

# obtained from Wikipedia Miner
# https://github.com/studio-ousia/wikipedia-miner/blob/master/src/org/wikipedia/miner/extraction/LanguageConfiguration.java
REDIRECT_REGEXP = re.compile(
    ur"(?:\#|＃)(?:REDIRECT|転送)[:\s]*(?:\[\[(.*)\]\]|(.*))", re.IGNORECASE
)


cdef class Word:
    def __init__(self, unicode text):
        self.text = text

    def __repr__(self):
        return '<Word %s>' % (self.text.encode('utf-8'))

    def __reduce__(self):
        return (self.__class__, (self.text,))


cdef class WikiLink:
    def __init__(self, unicode title, unicode text, list words):
        self.title = title
        self.text = text
        self.words = words

    def __repr__(self):
        return '<WikiLink %s>' % (self.title.encode('utf-8'))

    def __reduce__(self):
        return (self.__class__, (self.title, self.text, self.words))


cdef class WikiPage:
    def __init__(self, title, language, wiki_text):
        self.title = title
        self.language = language
        self.wiki_text = wiki_text

    def __repr__(self):
        return '<WikiPage %s>' % (self.title.encode('utf-8'))

    def __reduce__(self):
        return (self.__class__, (self.title, self.language, self.wiki_text))

    @property
    def is_redirect(self):
        return bool(self.redirect)

    @property
    def redirect(self):
        red_match_obj = REDIRECT_REGEXP.match(self.wiki_text)
        if not red_match_obj:
            return None

        if red_match_obj.group(1):
            dest = red_match_obj.group(1)
        else:
            dest = red_match_obj.group(2)

        return self._normalize_title(dest)

    def extract_paragraphs(self, int min_paragraph_len=20, bint phrase=False):
        cdef int n, char_start, char_end, token_start, token_end
        cdef unicode text, title, matched, link_text
        cdef tuple wikilink_obj
        cdef list text_arr, wikilinks_arr, tokens, mentions, words, paragraph,\
            paragraphs, prefixes
        cdef dict char_start_index, char_end_index, mention_index,\
            wikilink_index

        if self.is_redirect:
            return

        text_arr = [u'']
        wikilinks_arr = [[]]

        for node in self._parse().nodes:
            if isinstance(node, mwparserfromhell.nodes.Text):
                for (n, text) in enumerate(unicode(node).split('\n')):
                    if n == 0:
                        text_arr[-1] += text
                    else:
                        text_arr.append(text)
                        wikilinks_arr.append([])

            elif isinstance(node, mwparserfromhell.nodes.Wikilink):
                title = node.title.strip_code()
                if not title:
                    continue

                if node.text:
                    text = node.text.strip_code()
                else:
                    text = node.title.strip_code()

                char_start = len(text_arr[-1])
                char_end = char_start + len(text)

                text_arr[-1] += text
                wikilinks_arr[-1].append(
                    ((char_start, char_end), self._normalize_title(title), text)
                )

            elif isinstance(node, mwparserfromhell.nodes.Tag):
                # include only bold or italic tags
                if node.tag not in ('b', 'i'):
                    continue
                if not node.contents:
                    continue

                text_arr[-1] += node.contents.strip_code()

        tokenizer = _get_tokenizer(self.language)
        if phrase:
            ner = _get_ner(self.language)

        paragraphs = []

        for (text, wikilinks) in zip(text_arr, wikilinks_arr):
            tokens = tokenizer.tokenize(text)

            char_start_index = {t.span[0]: n for (n, t) in enumerate(tokens)}
            char_end_index = {t.span[1]: n for (n, t) in enumerate(tokens)}

            wikilink_index = {}
            for (n, wikilink_obj) in enumerate(wikilinks):
                (char_start, char_end) = wikilink_obj[0]
                if (
                    char_start in char_start_index and
                    char_end in char_end_index
                ):
                    token_start = char_start_index[char_start]
                    wikilink_index[token_start] = wikilink_obj

            if phrase:
                mentions = ner.extract([t.text for t in tokens])
                mention_index = {m.span[0]: m for (n, m) in enumerate(mentions)}

            cur = 0
            paragraph = []
            for (n, token) in enumerate(tokens):
                if n < cur:
                    continue

                if n in wikilink_index:
                    ((char_start, char_end), title, link_text) = wikilink_index[n]
                    token_start = char_start_index[char_start]
                    token_end = char_end_index[char_end] + 1

                    if phrase:
                        words = [Word(link_text)]
                    else:
                        words = [Word(t.text)
                                 for t in tokens[token_start:token_end]]

                    paragraph.append(WikiLink(title, link_text, words))
                    cur = token_end

                elif phrase and n in mention_index:
                    (token_start, token_end) = mention_index[n].span
                    char_start = tokens[token_start].span[0]
                    char_end = tokens[token_end-1].span[1]

                    paragraph.append(Word(text[char_start:char_end]))
                    cur = token_end

                else:
                    paragraph.append(Word(token.text))

            paragraphs.append(paragraph)

        for paragraph in paragraphs:
            if (
                paragraph and
                (paragraph[0].text and (paragraph[0].text[0] not in ('|', '!', '{'))) and  # remove wikitables
                (len(paragraph) >= min_paragraph_len)  # remove paragraphs that are too short
            ):
                yield paragraph

    cdef _parse(self):
        try:
            return mwparserfromhell.parse(self.wiki_text)

        except Exception:
            logger.exception('Failed to parse wiki text: %s', self.title)
            return mwparserfromhell.parse('')

    cdef inline unicode _normalize_title(self, unicode title):
        title = title[0].upper() + title[1:]
        return title.replace('_', ' ')


@lru_cache(1)
def _get_tokenizer(language):
    from utils.tokenizer.opennlp import OpenNLPTokenizer
    from utils.tokenizer.mecab import MeCabTokenizer

    if language == 'en':
        return OpenNLPTokenizer()
    elif language == 'ja':
        return MeCabTokenizer()
    else:
        raise NotImplementedError('Unsupported language')


@lru_cache(1)
def _get_ner(language):
    from utils.ner.stanford_ner import StanfordNER

    if language == 'en':
        return StanfordNER()
    else:
        raise NotImplementedError('Unsupported language')
