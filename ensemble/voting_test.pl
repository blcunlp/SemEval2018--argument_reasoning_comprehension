use FindBin qw($Bin);
use utf8;

if(@ARGV!=1){
	print "usage: perl voting_test.pl dir_name\n";
    exit;
}

open(Ans,"<:encoding(utf-8)","../data/truth.txt") || die("cannot openfile $!");
while(<Ans>){
    if(/^(\S+)\s+(\d)\s+/){
        $hash_correct{$1} = $2;
    }
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

open(Out,">:encoding(utf-8)","test_voted_answer");
$right_num = 0;
foreach $ele(sort keys %hash){
    $score = $hash{$ele}/(@files-2);
    if($score > 0.5){
        print Out ($ele."\t1\n");
        if (defined $hash_correct{$ele} and $hash_correct{$ele}==1){
            $right_num += 1;
        }
    }
    else{
        print Out ($ele."\t0\n");
        if (defined $hash_correct{$ele} and $hash_correct{$ele}==0){
            $right_num += 1;
        }
    }
}
print("test answer saved in test_voted_answer\n");
if( keys %hash_correct >0){
    $acc = $right_num/(keys %hash_correct);
    $annote_num = keys %hash_correct;
}
print "right_num:$right_num\nannote_num: $annote_num\n";
print("acc after vote: $acc\n");
