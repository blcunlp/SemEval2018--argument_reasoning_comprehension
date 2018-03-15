use FindBin qw($Bin);
use utf8;

if(@ARGV!=1){
	print "usage: perl voting_dev.pl dev_dir \n";
    exit;
}

open(Ans,"<:encoding(utf-8)","../data/dev/dev-only-labels.txt") || die("cannot openfile $!");
while($line = <Ans>){
    @ele = split("\t",$line);
    $hash_correct{$ele[0]} = $ele[1];
}


opendir(Dir,"$Bin/$ARGV[0]") || die ("cannot open dir $!");
@files = readdir(Dir);
foreach $file (@files){
	open(In,"<:encoding(utf-8)","$Bin/$ARGV[0]/$file") || die ("cannot openfile $!");
	while($line = <In>){
        @ele = split("\t",$line);
        $hash{$ele[0]} = $hash{$ele[0]} + $ele[1];
	}
	close(In);
}
open(Out,">:encoding(utf-8)","$Bin/voted_res");
foreach $ele(sort keys %hash){
    $score = $hash{$ele}/(@files-2);
    if($score > 0.5){
        print Out ($ele."\t1\n");
        if ($hash_correct{$ele} == 1){
            $right_num += 1;
        }
    }
    else{
        print Out ($ele."\t0\n");
        if($hash_correct{$ele} == 0){
            $right_num += 1;
        }
    }
}

$acc = sprintf "%.8f",$right_num/(keys %hash);
print ("acc: $acc\n");
