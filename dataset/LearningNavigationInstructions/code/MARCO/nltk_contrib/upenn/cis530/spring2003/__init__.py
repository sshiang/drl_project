# Natural Language Toolkit: Third-Party Contributions
# Contributions from the University of Pennsylvania: CIS-530/spring2003
#
# Copyright (C) 2003 The original contributors
# URL: <http://nltk.sf.net>
#
# $Id: __init__.py,v 1.1 2003/08/07 05:24:12 edloper Exp $

"""
Student projects from the course CIS-530, taught in Spring of 2003 at
the University of Pennsylvania.
"""

# Add all subdirectories to our package contents path.  This lets us
# put modules in separate subdirectories without making them packages.
import nltk_contrib
nltk_contrib._add_subdirectories_to_package(__path__)
