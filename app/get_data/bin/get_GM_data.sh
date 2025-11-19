#!/usr/bin/bash -l

fcst_date=$1
datadir=$2

year=${fcst_date:0:4}
date=${fcst_date:0:8}
hour=${fcst_date:9:2}

### Determine lead times based on forecast hour ###

if [[ "$hour" == "00" || "$hour" == "12" ]]; then
    leads_6h_end='72'
    leads_24h_end='144'
else
    leads_6h_end='60'
    leads_24h_end='48'
fi

## Define times for global update analysis - only 000 and 006 available ##
analysis_runs=("000" "006")

### Get 6hr leads ###
leads_6h=""
for ((i=0; i<=leads_6h_end; i+=6)); do
    leads_6h="${leads_6h}${i} "
done
leads_6h=$(echo $leads_6h)  # Remove trailing space if needed

### Get 24hr leads ###
leads_24h=""
for ((i=0; i<=leads_24h_end; i+=24)); do
    leads_24h="${leads_24h}${i} "
done
leads_24h=$(echo $leads_24h)  # Remove trailing space if needed


#################### UPDATE ANALYSES ##########################

mass-pull () {
  local analysis=$1 
  touch query
  cat >query <<EOF
begin
 filename="prods_op_gl-up_${date}_${hour}_${analysis}.pp"
 stash=(5201,5202,4201,4202)
end
EOF

  moo select -I query moose:/opfc/atm/global/prods/${year}.pp/ ${datadir}/6_hour/${fcst_date}_gl-up_${analysis}.pp
  rm query
}

for analysis in "${analysis_runs[@]}"; do
  mass-pull "$analysis"
done

#################### 6 hrly ###########################

mass-pull () {
touch query
cat >query <<EOF
  begin
    filename="prods_op_gl-mn_${date}_${hour}_*.pp"
    stash=(5201,5202,4201,4202)
    lbft=${lead}
  end
EOF

moo select -I query moose:/opfc/atm/global/prods/${year}.pp/ ${datadir}/6_hour/${fcst_date}_gl-mn_T${this_lead}.pp

}


# Pull 18 member mogreps ensemble from MASS archive
list_of_files=''

if [ ! -d "${datadir}/6_hour" ] ; then
  echo "${datadir}/6_hour doesn't currently exist. Making..."
  mkdir -p ${datadir}/6_hour
fi

for lead in $leads_6h
do
  this_lead=$(printf "%03d" ${lead})
  echo ${this_lead}
  file_to_cat="${datadir}/6_hour/${fcst_date}_gl-mn_T${this_lead}.pp"
  list_of_files=$(echo ${list_of_files} "${file_to_cat} ")
  mass-pull
  rm query
done


accum='006'
echo "cat $list_of_files > ${datadir}/6_hour/${fcst_date}_gl-mn_${accum}.pp"
cat $list_of_files > ${datadir}/6_hour/${fcst_date}_gl-mn_${accum}.pp

##################### 24 hrly ##########################

mass-pull1 () {
touch query1
cat >query1 <<EOF
begin
 filename="prods_op_gl-mn_${date}_${hour}_*.pp"
 stash=(5201,5202,4201,4202)
 lbft=${lead}
end
EOF

moo select -I query1 moose:/opfc/atm/global/prods/${year}.pp/ ${datadir}/24_hour/${fcst_date}_gl-mn_T${this_lead}.pp

}
list_of_files=''

if [ ! -d "${datadir}/24_hour" ] ; then
  echo "${datadir}/24_hour doesn't currently exist. Making..."
  mkdir -p ${datadir}/24_hour
fi

for lead in $leads_24h
do
  this_lead=$(printf "%03d" ${lead})
  echo ${this_lead}
  file_to_cat="${datadir}/24_hour/${fcst_date}_gl-mn_T${this_lead}.pp"
  list_of_files=$(echo ${list_of_files} "${file_to_cat} ")
  mass-pull1
  rm query1
done

accum='024'
echo "cat $list_of_files > ${datadir}/24_hour/${fcst_date}_gl-mn_${accum}.pp"
cat $list_of_files > ${datadir}/24_hour/${fcst_date}_gl-mn_${accum}.pp


############################## FOR TRIALS #########################################
# trial_name1=$2
# trial_name2=$3
# shortened_trial_name1=$(echo ${trial_name1} | tr -d '-')
# shortened_trial_name2=$(echo ${trial_name2} | tr -d '-')
# moo select query moose:/devfc/${trial_name1}/field.pp/ ${datadir}/${fcst_date}_${shortened_trial_name1}.pp
# moo select query moose:/devfc/${trial_name2}/field.pp/ ${datadir}/${fcst_date}_${shortened_trial_name2}.pp
