# -*- coding: utf-8 -*-

import pkg_resources
import unittest

from entity_vector.wiki_dump_reader import WikiDumpReader
from entity_vector.wiki_page import WikiPage

from nose.tools import *


class TestWikiDumpReader(unittest.TestCase):
    def setUp(self):
        sample_dump_file = pkg_resources.resource_filename(
            __name__, './test_data/enwiki-pages-articles-sample.xml.bz2'
        )
        self.dump_reader = WikiDumpReader(sample_dump_file)

    def test_language_property(self):
        eq_('en', self.dump_reader.language)

    def test_iterator(self):
        pages = list(self.dump_reader)
        eq_(3, len(pages))
        ok_(all([isinstance(page, WikiPage) for page in pages]))
