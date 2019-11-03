SUBDIRS=src
dist_doc_DATA=README

TESTSUITES_DIR = testsuites
EXECUTABLE=$(SUBDIRS)/reverse

testcases = nasm
           
.PHONY: all $(testcases)

$(testcases):%:
	$(EXECUTABLE) $(TESTSUITES_DIR)/$@/
