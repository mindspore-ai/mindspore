#!/usr/bin/env perl

use warnings;
use strict;

while(<STDIN>) {
  s/”/\&quot;/g;
  s/“/\&quot;/g;
  s/„/\&quot;/g;
  print $_;
}