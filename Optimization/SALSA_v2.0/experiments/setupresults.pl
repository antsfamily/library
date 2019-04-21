#!/usr/bin/perl

$curr_dir = `chdir`;
chomp($curr_dir);

@levels = split(/\\/, $curr_dir);
$size = scalar @levels;

$workingdir = $levels[$size-1];
print $workingdir;

#exec('cd ../Results/$workingdir');
##exec('cd Results');
#exec('mkdir $workingdir');
#
##$mainpath = join(/\\/,);