# Audit trace: raw records vs derived artifacts

Seed 620, 20 random samples. Result: **PASS**.

## Feature cells (raw Parquet cell and name dictionary vs X.npz and feature catalog)

| sample_id | table | feature_id | raw_name | expect | raw_value | X_value | name_matches | ok |
|---|---|---|---|---|---|---|---|---|
| 10381 | 5_str | 5288795 | STRING:messageboxa | 1 | 1 | 1 | True | True |
| 10381 | 7_sys | 6400737 | SYSTEM:COMMAND_LINE:"c:\program files\vs revo group\revo uninstaller pro\ruplp.e | 0 | 0 | 0 | True | True |
| 10626 | 1_api | 53 | API:RegCloseKey | 1 | 1 | 1 | True | True |
| 10626 | 4_dir | 2629599 | DIRECTORY:CREATED:c:\users\administrator\appdata\local\temp\mia88a8.tmp\data\off | 0 | 0 | 0 | True | True |
| 10754 | 1_api | 53 | API:RegCloseKey | 1 | 1 | 1 | True | True |
| 10754 | 2_reg | 491409 | REG:WRITTEN:HKEY_LOCAL_MACHINE\SOFTWARE\Classes\Wow6432Node\Interface\{82351440- | 0 | 0 | 0 | True | True |
| 11437 | 4_dir | 2684944 | DIRECTORY:ENUMERATED:c:\users\administrator\appdata\local\temp\0ns\*.exe | 1 | 1 | 1 | True | True |
| 11437 | 6_net | 6395028 | NETWORK:CONNECTS_HOST:stats.securebrowser.com | 0 | 0 | 0 | True | True |
| 11846 | 3_file | 1683996 | FILE:EXISTS:c:\users\administrator\appdata\local\temp\cmd.bat | 1 | 1 | 1 | True | True |
| 11846 | 3_file | 1771570 | FILE:EXISTS:c:\users\administrator\appdata\roaming\avs4you\avsaudioconverter.dll | 0 | 0 | 0 | True | True |
| 11973 | 2_reg | 418681 | REG:READ:HKEY_LOCAL_MACHINE\SYSTEM\ControlSet001\Control\WaitToKillServiceTimeou | 1 | 1 | 1 | True | True |
| 11973 | 8_drop | 6416466 | DROP:EXTENSION:01ldjg | 0 | 0 | 0 | True | True |
| 12013 | 1_api | 10 | API:NtOpenDirectoryObject | 1 | 1 | 1 | True | True |
| 12013 | 1_api | 279 | API:InternetReadFile | 0 | 0 | 0 | True | True |
| 12446 | 9_sig | 6468070 | SIGNATURE:allocates_rwx | 1 | 1 | 1 | True | True |
| 12446 | 4_dir | 2743957 | DIRECTORY:ENUMERATED:c:\windows\winsxs\amd64_microsoft-windows-s..restartup-repa | 0 | 0 | 0 | True | True |
| 13159 | 3_file | 1789843 | FILE:EXISTS:c:\warthunder\content\pkg_dev\base.vromfs.bin | 1 | 1 | 1 | True | True |
| 13159 | 5_str | 4596507 | STRING:dbbbbbbbn | 0 | 0 | 0 | True | True |
| 13168 | 7_sys | 6412694 | SYSTEM:DLL_LOADED:c:\windows\winsxs\amd64_microsoft.windows.gdiplus_6595b64144cc | 1 | 1 | 1 | True | True |
| 13168 | 7_sys | 6411660 | SYSTEM:DLL_LOADED:c:\users\admini~1\appdata\local\temp\{6bf90d91-c5db-454e-a7b4- | 0 | 0 | 0 | True | True |
| 13346 | 7_sys | 6414128 | SYSTEM:GUID:{ab310581-ac80-11d1-8df3-00c04fb6ef69} | 1 | 1 | 1 | True | True |
| 13346 | 4_dir | 2626418 | DIRECTORY:CREATED:c:\users\administrator\appdata\local\temp\7zs8a9c.tmp\zh-hant | 0 | 0 | 0 | True | True |
| 13719 | 6_net | 6399244 | NETWORK:RESOLVES_HOST:s1.symcb.com | 1 | 1 | 1 | True | True |
| 13719 | 3_file | 1252027 | FILE:EXISTS:c:\program files (x86)\moviepanda\moviepanda\lib\django\contrib\admi | 0 | 0 | 0 | True | True |
| 13783 | 9_sig | 6468267 | SIGNATURE:uses_windows_utilities | 1 | 1 | 1 | True | True |
| 13783 | 3_file | 1603587 | FILE:EXISTS:c:\python27\lib\test\decimaltestdata\minmag.dectest.id[501556fd-3483 | 0 | 0 | 0 | True | True |
| 13990 | 2_reg | 100821 | REG:OPENED:HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\CTF\TIP\{A028AE76-01B1-46C2-99C | 1 | 1 | 1 | True | True |
| 13990 | 1_api | 239 | API:WriteConsoleA | 0 | 0 | 0 | True | True |
| 14078 | 2_reg | 112774 | REG:OPENED:HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Keyboard Layouts\ | 1 | 1 | 1 | True | True |
| 14078 | 2_reg | 285852 | REG:READ:HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Setup\PnpL | 0 | 0 | 0 | True | True |
| 14279 | 5_str | 6013934 | STRING:w68> [ | 1 | 1 | 1 | True | True |
| 14279 | 2_reg | 403494 | REG:READ:HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersio | 0 | 0 | 0 | True | True |
| 14949 | 5_str | 3191318 | STRING:);mhnm | 1 | 1 | 1 | True | True |
| 14949 | 7_sys | 6407994 | SYSTEM:DLL_LOADED:c:\program files (x86)\common files\avsmedia\activex\avsimages | 0 | 0 | 0 | True | True |
| 15266 | 7_sys | 6412360 | SYSTEM:DLL_LOADED:c:\windows\system32\riched20.dll | 1 | 1 | 1 | True | True |
| 15266 | 5_str | 5957811 | STRING:variantclear | 0 | 0 | 0 | True | True |
| 15643 | 3_file | 1729340 | FILE:EXISTS:c:\users\administrator\appdata\local\temp\rarsfx0\release\databases\ | 1 | 1 | 1 | True | True |
| 15643 | 8_drop | 6428137 | DROP:EXTENSION:i55 | 0 | 0 | 0 | True | True |
| 15666 | 8_drop | 6457239 | DROP:TYPE:pe32_executable_(dll)_(gui)_intel_80386,_for_ms_windows | 1 | 1 | 1 | True | True |
| 15666 | 1_api | 177 | API:AssignProcessToJobObject | 0 | 0 | 0 | True | True |

## <SAMPLE>-normalised features (derived cell vs raw cell naming the own sample ID)

| sample_id | table | derived_name | raw_source_found | raw_value | ok |
|---|---|---|---|---|---|
| 10381 | 3_file | FILE:READ:c:\users\administrator\appdata\local\temp\<SAMPLE>.exe | True | 1 | True |
| 10626 | 3_file | FILE:EXISTS:c:\users\administrator\appdata\local\temp\<SAMPLE>.exe | True | 1 | True |
| 10754 | 3_file | FILE:OPENED:c:\users\administrator\appdata\local\temp\<SAMPLE>.exe | True | 1 | True |
| 11437 | 3_file | FILE:OPENED:c:\users\administrator\appdata\local\temp\<SAMPLE>.exe | True | 1 | True |
| 11846 | 4_dir | DIRECTORY:ENUMERATED:c:\users\administrator\appdata\local\temp\<SAMPLE>.exe | True | 1 | True |
| 12013 | 7_sys | SYSTEM:COMMAND_LINE:c:\users\administrator\appdata\local\temp\<SAMPLE> | True | 1 | True |
| 13159 | 7_sys | SYSTEM:COMMAND_LINE:"c:\users\admini~1\appdata\local\temp\is-6jq51.tmp\<SAMPLE>. | True | 1 | True |
| 13168 | 3_file | FILE:OPENED:c:\users\administrator\appdata\local\temp\<SAMPLE>.exe.html | True | 1 | True |
| 13346 | 3_file | FILE:OPENED:c:\users\administrator\appdata\local\temp\<SAMPLE>.zip.html | True | 1 | True |
| 13719 | 3_file | FILE:EXISTS:c:\users\administrator\appdata\local\temp\is-800kr.tmp\<SAMPLE>.tmp | True | 1 | True |
| 13783 | 3_file | FILE:OPENED:c:\users\administrator\appdata\local\temp\<SAMPLE>.exe.html | True | 1 | True |
| 13990 | 3_file | FILE:OPENED:c:\users\administrator\appdata\local\temp\<SAMPLE>.msi | True | 1 | True |
| 14078 | 3_file | FILE:OPENED:c:\users\administrator\appdata\local\temp\<SAMPLE>.msi | True | 1 | True |
| 15266 | 2_reg | REG:OPENED:HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\CTF\Compatibility\<SAMPLE>.exe | True | 1 | True |
| 15643 | 3_file | FILE:OPENED:c:\users\administrator\appdata\local\temp\<SAMPLE>.exe | True | 1 | True |
| 15666 | 3_file | FILE:OPENED:c:\users\administrator\appdata\local\temp\<SAMPLE>.msi | True | 1 | True |

## Labels and aggregates (raw metadata and raw columns vs labels.csv and aggregates)

| sample_id | raw_label | label | raw_family | family | raw_api_n | agg_api_n | raw_sig_n | agg_sig_n | ok |
|---|---|---|---|---|---|---|---|---|---|
| 10381 | 1 | 1 | pdpcoder | pdpcoder | 60 | 60 | 14 | 14 | True |
| 10626 | 1 | 1 | cerber | cerber | 102 | 102 | 22 | 22 | True |
| 10754 | 1 | 1 | dharma | dharma | 82 | 82 | 17 | 17 | True |
| 11437 | 1 | 1 | pajetbin | pajetbin | 49 | 49 | 9 | 9 | True |
| 11846 | 1 | 1 | bianlian | bianlian | 60 | 60 | 10 | 10 | True |
| 11973 | 1 | 1 | hive | hive | 44 | 44 | 6 | 6 | True |
| 12013 | 1 | 1 | vhd | vhd | 96 | 96 | 9 | 9 | True |
| 12446 | 1 | 1 | 8base | 8base | 8 | 8 | 4 | 4 | True |
| 13159 | 0 | 0 | goodware | goodware | 164 | 164 | 21 | 21 | True |
| 13168 | 0 | 0 | goodware | goodware | 17 | 17 | 3 | 3 | True |
| 13346 | 0 | 0 | goodware | goodware | 16 | 16 | 3 | 3 | True |
| 13719 | 0 | 0 | goodware | goodware | 133 | 133 | 22 | 22 | True |
| 13783 | 0 | 0 | goodware | goodware | 16 | 16 | 4 | 4 | True |
| 13990 | 0 | 0 | goodware | goodware | 78 | 78 | 7 | 7 | True |
| 14078 | 0 | 0 | goodware | goodware | 102 | 102 | 10 | 10 | True |
| 14279 | 0 | 0 | goodware | goodware | 0 | 0 | 0 | 0 | True |
| 14949 | 0 | 0 | goodware | goodware | 0 | 0 | 0 | 0 | True |
| 15266 | 0 | 0 | goodware | goodware | 67 | 67 | 5 | 5 | True |
| 15643 | 0 | 0 | goodware | goodware | 116 | 116 | 12 | 12 | True |
| 15666 | 0 | 0 | goodware | goodware | 105 | 105 | 11 | 11 | True |
