----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 08/29/2019 07:35:14 PM
-- Design Name: 
-- Module Name: wrapper_EB_DEC - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;




entity PE_wrapper_EB_DEC is
generic(
    x_init          : integer := 0;
    y_init          : integer := 0;
    img_width       : natural;
    img_height      : natural;
    n_frames        : natural;
    n_steps         : natural;
    pix_depth       : natural;
    subimg_width    : natural;
    subimg_height   : natural;
    buffer_length   : natural
);
  Port (
clk                        : in std_logic ;
reset                        : in std_logic ;
--reset_in_pe_ctrl            : in std_logic;
--------Handshakes da entrada do EB
--P_PE_valid_CF_EB              : in    std_logic; -- indica msg dispon?vel pra enviar ao EB
P_PE_ready_EB_CF              : out   std_logic; -- indica que o EB pode receber a mensagem 
P_PE_stall_in_EB              : in    std_logic ; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF
P_PE_stall_out_EB             : inout std_logic;  -- indica que o '''
---------handshakes na sa?da do decoder
P_PE_out_send                 : out  std_logic;
--P_PE_ready_CF_DEC             : in std_logic;
P_PE_stall_out_DEC            : inout std_logic ;
P_PE_stall_in_DEC             : in std_logic ;
-------------------------------------------
--P_PE_in_ack : in std_logic;

P_PE_data_in_pixel     : in std_logic_vector(pix_depth-1 downto 0);
P_PE_data_in_x_dest    : in std_logic_vector(img_width-1 downto 0);
P_PE_data_in_y_dest    : in std_logic_vector(img_height-1 downto 0);
P_PE_data_in_step      : in std_logic_vector(n_steps-1 downto 0);
P_PE_data_in_frame     : in std_logic_vector(n_frames-1 downto 0);
P_PE_data_in_x_orig    : in std_logic_vector(img_width-1 downto 0);
P_PE_data_in_y_orig    : in std_logic_vector(img_height-1 downto 0);
P_PE_data_in_fb        : in std_logic;
 
P_PE_data_out_pixel     : out std_logic_vector(pix_depth-1 downto 0);    
P_PE_data_out_x_dest    : out std_logic_vector(img_width-1 downto 0);    
P_PE_data_out_y_dest    : out std_logic_vector(img_height-1 downto 0);   
P_PE_data_out_step      : out std_logic_vector(n_steps-1 downto 0);      
P_PE_data_out_frame     : out std_logic_vector(n_frames-1 downto 0);     
P_PE_data_out_x_orig    : out std_logic_vector(img_width-1 downto 0);    
P_PE_data_out_y_orig    : out std_logic_vector(img_height-1 downto 0);   
P_PE_data_out_fb        : out std_logic   ;
--Wrapper_PE_ready_EB_CF  : in std_logic;
P_PE_out_direction      : out std_logic_vector(5 downto 0);
P_PE_ack                : out std_logic;
P_PE_req                : in std_logic
   
  
   );
end PE_wrapper_EB_DEC;

architecture Behavioral of PE_wrapper_EB_DEC is


signal	   w_PE_ready_DEC_EB             :  std_logic; -- sinal indica se o decoder est? pronto para receber a mensagem 

signal     w_PE_x_dest                   :  std_logic_vector(img_width-1 downto 0);
signal     w_PE_y_dest                   :  std_logic_vector(img_height-1 downto 0);
signal     w_PE_x_orig                   :  std_logic_vector(img_width-1 downto 0);
signal     w_PE_y_orig                   :  std_logic_vector(img_height-1 downto 0);
signal     w_PE_fb                       :  std_logic;
signal     w_PE_out_send                 :  std_logic;
signal     w_PE_ready_CF_DEC             :  std_logic;
signal     w_PE_out_direction            :  std_logic_vector(5 downto 0);

SIGnal     w_PE_valid_EB_DEC   :  std_logic;
SIGnal     w_PE_out_new_EB_DEC  :  std_logic;

signal    s_PE_data_pixel     :  std_logic_vector(pix_depth-1 downto 0);  
signal    s_PE_data_x_dest    :  std_logic_vector(img_width-1 downto 0);  
signal    s_PE_data_y_dest    :  std_logic_vector(img_height-1 downto 0); 
signal    s_PE_data_step      :  std_logic_vector(n_steps-1 downto 0);    
signal    s_PE_data_frame     :  std_logic_vector(n_frames-1 downto 0);   
signal    s_PE_data_x_orig    :  std_logic_vector(img_width-1 downto 0);  
signal    s_PE_data_y_orig    :  std_logic_vector(img_height-1 downto 0); 
signal    s_PE_data_fb        :  std_logic   ;                                                                    
signal    s_PE_ack                :  std_logic;                               
signal    s_PE_req                : std_logic;                              





      
begin

Inst_dec : entity work.PE_dec

generic map(
                                    x_init          => x_init       ,
                                    y_init          => y_init       ,
                                    img_width       => img_width    ,
                                    img_height      => img_height   ,
                                    n_frames        => n_frames     ,
                                    n_steps         => n_steps      ,
                                    pix_depth       => pix_depth    ,
                                    subimg_width    => subimg_width ,
                                    subimg_height   => subimg_height,
                                    buffer_length   => buffer_length
                                    )

port map (


clk                   =>  clk ,
rst                   => reset ,



PE_stall_out_DEC      =>   P_PE_stall_out_DEC ,
PE_stall_in_DEC       =>   P_PE_stall_in_DEC  ,

PE_in_x_dest          =>  w_PE_x_dest,
PE_in_y_dest          =>  w_PE_y_dest,
PE_in_x_orig          =>  w_PE_x_orig,
PE_in_y_orig          =>  w_PE_y_orig,
PE_in_fb              =>  w_PE_fb    ,

PE_out_direction      =>  P_PE_out_direction,

PE_out_send           => P_PE_out_send   ,


PE_ready_DEC_EB       => w_PE_ready_DEC_EB,
PE_valid_EB_DEC       => w_PE_valid_EB_DEC,
PE_out_new_EB_DEC     => w_PE_out_new_EB_DEC

);

Inst_elastic_buffer : entity work.PE_elastic_buffer

generic map(
x_init          => x_init       ,
y_init          => y_init       ,
img_width       => img_width    ,
img_height      => img_height   ,
n_frames        => n_frames     ,
n_steps         => n_steps      ,
pix_depth       => pix_depth    ,
subimg_width    => subimg_width ,
subimg_height   => subimg_height,
buffer_length   => buffer_length
)
port map (


clk                     =>     clk , 
rst                     =>     reset ,       
          
PE_valid_EB_DEC         =>     w_PE_valid_EB_DEC,
PE_ready_DEC_EB         =>     w_PE_ready_DEC_EB,
  
PE_out_new_EB_DEC     => w_PE_out_new_EB_DEC,  

PE_ready_EB_CF          =>  P_PE_ready_EB_CF,
    
PE_stall_in_EB          => P_PE_stall_in_EB,
PE_stall_out_EB         => P_PE_stall_out_EB,

                       
PE_data_in_pixel        =>    s_PE_data_pixel  ,
PE_data_in_x_dest       =>    s_PE_data_x_dest ,
PE_data_in_y_dest       =>    s_PE_data_y_dest ,
PE_data_in_step         =>    s_PE_data_step   ,
PE_data_in_frame        =>    s_PE_data_frame  ,
PE_data_in_x_orig       =>    s_PE_data_x_orig ,
PE_data_in_y_orig       =>    s_PE_data_y_orig ,
PE_data_in_fb           =>    s_PE_data_fb     ,
PE_ack                  =>    s_PE_ack         ,
PE_req                  =>    s_PE_req         ,

PE_data_out_pixel       =>     P_PE_data_out_pixel ,
PE_data_out_x_dest      =>     w_PE_x_dest,
PE_data_out_y_dest      =>     w_PE_y_dest,
PE_data_out_step        =>     P_PE_data_out_step  ,
PE_data_out_frame       =>     P_PE_data_out_frame ,
PE_data_out_x_orig      =>     w_PE_x_orig,
PE_data_out_y_orig      =>     w_PE_y_orig,
PE_data_out_fb          =>     w_PE_fb  
--PE_ready_CF_EB          =>     Wrapper_PE_ready_EB_CF,






   


);


Inst_in_PE_ctrl : entity work.in_PE_controller

generic map(
x_init          => x_init       ,
y_init          => y_init       ,
img_width       => img_width    ,
img_height      => img_height   ,
n_frames        => n_frames     ,
n_steps         => n_steps      ,
pix_depth       => pix_depth    
)
port map (


clk                     =>     clk, 
reset                   =>     reset,       
                                 
oc_PE_pixel        =>    P_PE_data_in_pixel  ,
oc_PE_x_dest       =>    P_PE_data_in_x_dest ,
oc_PE_y_dest       =>    P_PE_data_in_y_dest ,
oc_PE_step         =>    P_PE_data_in_step   ,
oc_PE_frame        =>    P_PE_data_in_frame  ,
oc_PE_x_orig       =>    P_PE_data_in_x_orig,
oc_PE_y_orig       =>    P_PE_data_in_y_orig,
oc_PE_fb           =>    P_PE_data_in_fb    ,        
oc_PE_new_msg      =>    P_PE_req,
oc_PE_ack          =>    P_PE_ack, 


    

 i_PE_pixel       =>     s_PE_data_pixel  ,
 i_PE_x_dest      =>     s_PE_data_x_dest ,
 i_PE_y_dest      =>     s_PE_data_y_dest ,
 i_PE_step        =>     s_PE_data_step   ,
 i_PE_frame       =>     s_PE_data_frame  ,
 i_PE_x_orig      =>     s_PE_data_x_orig ,
 i_PE_y_orig      =>     s_PE_data_y_orig ,
 i_PE_fb          =>     s_PE_data_fb     ,
 i_PE_req         =>     s_PE_req         ,
 i_PE_ack         =>     s_PE_ack          




   


);







     P_PE_data_out_x_dest <= w_PE_x_dest;
     P_PE_data_out_y_dest <= w_PE_y_dest;
     
     P_PE_data_out_x_orig <= w_PE_x_orig;
     P_PE_data_out_y_orig <= w_PE_y_orig;
     P_PE_data_out_fb     <= w_PE_fb    ;
     
 

   
end Behavioral;
