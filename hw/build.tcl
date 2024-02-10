
#!/usr/bin/env tclsh
#set x 10
#set y 3
#
## Perform arithmetic operations
#
#
#set xtile [expr [lindex $argv 0]]
#set ytile [expr [lindex $argv 1]]
#
#set subimg_width  [expr [lindex $argv 2]]
#set subimg_height [expr [lindex $argv 3]]
#
#puts "xtile: $xtile "
#puts "ytile: $ytile "
#puts "subimg_width: $subimg_width "
#puts "subimg_width: $subimg_width "

# Get the directory of the current script


#source /Users/arthu/VeryNewGitMPRISCV/hw/build.tcl


set script_path [info script]
set script_dir [file dirname $script_path]
set hw_dir "${script_dir}"
file mkdir "${hw_dir}/prjs"

for {set init 3} { $init < 7} { incr init } {





set xtiles [expr $init]
set ytiles [expr $init]



set project_dir "${hw_dir}/prjs/${xtiles}_${ytiles}"

# Check if the directory exists
if {[file exists $project_dir]} {
    # If the directory exists, delete it and its contents
    file delete -force $project_dir
    puts "Existing project directory $project_dir has been deleted."
}

# Proceed to create the project in the now-deleted (or never existing) directory
eval "create_project ${xtiles}_${ytiles} ${hw_dir}/prjs/${xtiles}_${ytiles}"

# -force -part xilinx.com:kv260\_som:part0:1.4"
set_property board_part xilinx.com:zcu104:part0:1.1 [current_project]

#open_project mpriscv.xpr
create_bd_design "MPRISCV"

set subimg_height [expr {ceil(240 / $ytiles)}]
set subimg_width [expr {ceil(240 / $xtiles)}]


set ADDR_WIDTH 10
set DATA_WIDTH 9
set MEM_WORDS 5000
set a_frames 1
set a_steps 3
set buffer_length 5
set img_height 8
set img_height_size 240
set img_width 8
set img_width_size 240
set n_frames 8
set n_steps 5
set pix_depth 16
set x_init 0
set x_local 0
set x_tiles $xtiles
set y_init 0
set y_local 0
set y_tiles $ytiles

# Adjust the paths to use the absolute path to the 'sources' directory
set sources_dir [file join $hw_dir sources]

# Add files from the 'sources' directory
add_files $sources_dir

# Add constraints file, adjust the path as needed
set constraints_file [file join $sources_dir constraints zcu104rev1.xdc]
add_files -fileset constrs_1 -norecurse $constraints_file



set_property file_type {VHDL 2008} [get_files ${hw_dir}/sources/tile/router/inputs/fifo_buffers/PM_elastic_buffer.vhd]
set_property file_type {VHDL 2008} [get_files ${hw_dir}/sources/tile/router/inputs/fifo_buffers/PE_elastic_buffer.vhd]
set_property file_type {VHDL 2008} [get_files ${hw_dir}/sources/tile/router/inputs/fifo_buffers/S_elastic_buffer.vhd]
set_property file_type {VHDL 2008} [get_files ${hw_dir}/sources/tile/router/inputs/fifo_buffers/N_elastic_buffer.vhd]
set_property file_type {VHDL 2008} [get_files ${hw_dir}/sources/tile/router/inputs/fifo_buffers/E_elastic_buffer.vhd]
set_property file_type {VHDL 2008} [get_files ${hw_dir}/sources/tile/router/inputs/fifo_buffers/W_elastic_buffer.vhd]


startgroup
create_bd_cell -type module -reference riscv2arm_wrapper riscv2arm_wrapper
endgroup

startgroup
create_bd_cell -type module -reference arm2riscv_wrapper arm2riscv_wrapper
endgroup

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat
endgroup

eval "set\_property CONFIG.NUM\_PORTS \{[expr $xtiles * $ytiles]\} \[get\_bd\_cells xlconcat] "

for {set hid 0} { $hid < $ytiles} { incr hid } {
    for {set wid 0} { $wid < $xtiles } { incr wid } {

      create_bd_cell -type module -reference tile  tile_$wid\_$hid
      set x_local [expr $wid]
      set y_local [expr $hid]
      set x_init  [expr $wid * $subimg_width]
      set y_init  [expr $hid * $subimg_height]

      eval "set\_property location \{[expr 1 + $wid] 300 [expr 300 * $hid]\} \[get\_bd\_cells tile\_$wid\_$hid\] "
      eval "set\_property CONFIG.ADDR_WIDTH \{[expr $ADDR_WIDTH]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.DATA_WIDTH \{[expr $DATA_WIDTH]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.MEM_WORDS \{[expr $MEM_WORDS]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.a_frames \{[expr $a_frames]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.a_steps \{[expr $a_steps]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.buffer_length \{[expr $buffer_length]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.img_height \{[expr $img_height]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.img_height_size \{[expr $img_height_size]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.img_width \{[expr $img_width]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.img_width_size \{[expr $img_width_size]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.n_frames \{[expr $n_frames]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.n_steps \{[expr $n_steps]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.pix_depth \{[expr $pix_depth]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.subimg_height \{[expr $subimg_height]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.subimg_width \{[expr $subimg_width]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.x_init \{[expr $x_init]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.x_local \{[expr $x_local]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.x_tiles \{[expr $x_tiles]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.y_init \{[expr $y_init]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.y_local \{[expr $y_local]\} \[get\_bd\_cells tile\_$wid\_$hid]"
      eval "set\_property CONFIG.y_tiles \{[expr $y_tiles]\} \[get\_bd\_cells tile\_$wid\_$hid]"
    }
}




for {set hid 0} { $hid < $ytiles} { incr hid } {
    for {set wid 0} { $wid < $xtiles } { incr wid } {

      if {$wid >= 0 && [expr $wid + 1] < $xtiles} {
      eval "connect\_bd\_net \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid]\/OUT\_E\] \[get\_bd\_pins tile\_[expr $wid+1]\_[expr $hid]\/IN\_W\]"
      eval "connect\_bd\_net \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid]\/OUT\_E\_ACK\] \[get\_bd\_pins tile\_[expr $wid+1]\_[expr $hid]\/IN\_W\_ACK\]"
      eval "connect\_bd\_net \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid]\/IN\_E\] \[get\_bd\_pins tile\_[expr $wid+1]\_[expr $hid]\/OUT\_W\]"
      eval "connect\_bd\_net \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid]\/IN\_E\_ACK\] \[get\_bd\_pins tile\_[expr $wid+1]\_[expr $hid]\/OUT\_W\_ACK\]"
      }
      if {$hid >= 0 && [expr $hid + 1] < $ytiles} {
      eval "connect\_bd\_net \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid]\/OUT\_S\] \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid+1]\/IN\_N\]"
      eval "connect\_bd\_net \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid]\/OUT\_S\_ACK\] \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid+1]\/IN\_N\_ACK\]"
      eval "connect\_bd\_net \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid]\/IN\_S\] \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid+1]\/OUT\_N\]"
      eval "connect\_bd\_net \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid]\/IN\_S\_ACK\] \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid+1]\/OUT\_N\_ACK\]"
      }


      eval "connect\_bd\_net \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid]\/saida\_init\_prog\_fim\] \[get\_bd\_pins xlconcat\/In[expr $wid + $ytiles * $hid]\]"
      eval "connect\_bd\_net \[get\_bd\_pins tile\_[expr $wid]\_[expr $hid]\/entrada\_init\_prog\_fim\] \[get\_bd\_pins xlconcat\/dout\]"

    }
}

connect_bd_net [get_bd_pins arm2riscv_wrapper/AXI_IMAGE_PIXEL] [get_bd_pins tile_1_1/axi_image_pixel]
connect_bd_net [get_bd_pins arm2riscv_wrapper/AXI_IMAGE_X] [get_bd_pins tile_1_1/axi_image_x]
connect_bd_net [get_bd_pins arm2riscv_wrapper/AXI_IMAGE_Y] [get_bd_pins tile_1_1/axi_image_y]
connect_bd_net [get_bd_pins arm2riscv_wrapper/AXI_IMAGE_REQ] [get_bd_pins tile_1_1/axi_image_req]
connect_bd_net [get_bd_pins arm2riscv_wrapper/AXI_IMAGE_ACK] [get_bd_pins tile_1_1/axi_image_ack]


connect_bd_net [get_bd_pins riscv2arm_wrapper/AXI_RISCV2ARM_PIXEL] [get_bd_pins tile_1_1/axi_riscv2arm_pixel]
connect_bd_net [get_bd_pins riscv2arm_wrapper/AXI_RISCV2ARM_REQ] [get_bd_pins tile_1_1/axi_riscv2arm_req]
connect_bd_net [get_bd_pins riscv2arm_wrapper/AXI_RISCV2ARM_ACK] [get_bd_pins tile_1_1/axi_riscv2arm_ack]

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq
endgroup

set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
] [get_bd_cells zynq]


set_property CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} [get_bd_cells zynq]



update_compile_order -fileset sources_1


startgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq/M_AXI_HPM0_FPD} Slave {/arm2riscv_wrapper/s00_axi} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins arm2riscv_wrapper/s00_axi]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq/M_AXI_HPM0_FPD} Slave {/riscv2arm_wrapper/s00_axi} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins riscv2arm_wrapper/s00_axi]

for {set hid 0} { $hid < $ytiles} { incr hid } {
    for {set wid 0} { $wid < $xtiles} { incr wid } {

      eval "apply\_bd\_automation -rule xilinx.com:bd\_rule:axi4 -config \{ Clk\_master \{Auto\} Clk\_slave \{Auto\} Clk\_xbar \{Auto\} Master \{\/zynq/M\_AXI\_HPM0\_FPD\} Slave \{\/tile\_[expr $hid]\_[expr $wid]\/s00\_axi\} ddr\_seg \{Auto\} intc\_ip \{New AXI Interconnect\} master\_apm \{0\}\}  \[get\_bd\_intf\_pins tile\_[expr $hid]\_[expr $wid]\/s00\_axi\]"
      eval "apply\_bd\_automation -rule xilinx.com:bd\_rule:clkrst -config \{ Clk \{\/zynq\/pl\_clk0 \(100 MHz\)\} Freq \{100\} Ref\_Clk0 \{\} Ref\_Clk1 \{\} Ref\_Clk2 \{\}\}  \[get\_bd\_pins tile\_[expr $hid]\_[expr $wid]\/s00\_axi\_aclk\]"
    }
}
endgroup



update_compile_order -fileset sources_1

apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq]

startgroup
set_property CONFIG.PSU__USE__M_AXI_GP1 {0} [get_bd_cells zynq]
delete_bd_objs [get_bd_intf_nets zynq_M_AXI_HPM1_FPD]
endgroup

update_compile_order -fileset sources_1

save_bd_design

# Assuming $script_dir is your script's directory and $xtiles, $ytiles are defined earlier in your script

# Construct the path for the BD file dynamically
set bd_file_path "${project_dir}/${xtiles}_${ytiles}.srcs/sources_1/bd/MPRISCV/MPRISCV.bd"

# Use eval to execute the make_wrapper command with the constructed path
make_wrapper -fileset [get_filesets sources_1] \
    -files [get_files $bd_file_path] -top


# Construct the path for the wrapper .v file dynamically
set wrapper_v_path "${project_dir}/${xtiles}_${ytiles}.gen/sources_1/bd/MPRISCV/hdl/MPRISCV_wrapper.v"

# Use eval to execute the add_files command with the constructed path for the wrapper .v file
eval "add_files -norecurse $wrapper_v_path"

eval "update\_compile\_order -fileset sources\_1"



set_property top MPRISCV_wrapper [current_fileset]
update_compile_order -fileset sources_1

set_property top MPRISCV_wrapper [current_fileset]
update_compile_order -fileset sources_1

save_bd_design

launch_runs impl_1 -to_step write_bitstream -jobs 2

# Wait until the runs are completed
wait_on_run impl_1
open_run impl_1


# Ensure the base configuration directory exists
set config_dir "$hw_dir/reports/${xtiles}_${ytiles}"
if {![file exists $config_dir]} {
    file mkdir $config_dir
    puts "Created configuration directory: $config_dir"
}

# Check if the directory exists
if {[file exists $config_dir]} {
    # Get a list of all items in the directory
    set items [glob -nocomplain -directory $config_dir *]

    # Iterate over each item and delete it
    foreach item $items {
        file delete -force -- $item
    }
    puts "All contents of $config_dir have been removed."
} else {
    puts "Directory $config_dir does not exist."
}

# Placeholders for report generation commands
# Ensure you replace these with actual commands to generate and save the reports
report_utilization -file "${config_dir}/utilization.rpt"
report_power -file "${config_dir}/power.rpt"


close_design

close_project


}