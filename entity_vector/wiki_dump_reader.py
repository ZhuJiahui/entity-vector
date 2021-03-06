# -*- coding: utf-8 -*-

import bz2
import logging
import re
from gensim.corpora import wikicorpus

from wiki_page import WikiPage

logger = logging.getLogger(__name__)

DEFAULT_IGNORED_NS = (
    'wikipedia:', 'category:', 'file:', 'portal:', 'template:', 'mediawiki:',
    'user:', 'help:', 'book:', 'draft:'
)


class WikiDumpReader(object):
    def __init__(self, dump_file, ignored_ns=DEFAULT_IGNORED_NS):
        self._dump_file = dump_file
        self._ignored_ns = ignored_ns

        with bz2.BZ2File(self._dump_file) as f:
            self._language = re.search(r'xml:lang="(.*)"', f.readline()).group(1)

    @property
    def language(self):
        return self._language

    def __iter__(self):
        with bz2.BZ2File(self._dump_file) as f:
            c = 0
            for (title, wiki_text, wiki_id) in wikicorpus.extract_pages(f):
                if any(
                    [title.lower().startswith(ns) for ns in self._ignored_ns]
                ):
                    continue
                c += 1

                yield WikiPage(
                    unicode(title), self._language, unicode(wiki_text)
                )

                if c % 10000 == 0:
                    logger.info('Processed: %d', c)
