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




entity E_wrapper_EB_DEC is
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
--------Handshakes da entrada do EB
--P_E_valid_CF_EB              : in    std_logic; -- indica msg dispon?vel pra enviar ao EB
P_E_ready_EB_CF              : out   std_logic; -- indica que o EB pode receber a mensagem 
P_E_stall_in_EB              : in    std_logic ; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF
P_E_stall_out_EB             : inout std_logic;  -- indica que o '''
---------handshakes na sa?da do decoder
P_E_out_send                 : out  std_logic;
--P_E_ready_CF_DEC             : in std_logic;
P_E_stall_out_DEC            : inout std_logic ;
P_E_stall_in_DEC             : in std_logic ;
-------------------------------------------
--P_E_in_ack : in std_logic;

P_E_data_in_pixel     : in std_logic_vector(pix_depth-1 downto 0);
P_E_data_in_x_dest    : in std_logic_vector(img_width-1 downto 0);
P_E_data_in_y_dest    : in std_logic_vector(img_height-1 downto 0);
P_E_data_in_step      : in std_logic_vector(n_steps-1 downto 0);
P_E_data_in_frame     : in std_logic_vector(n_frames-1 downto 0);
P_E_data_in_x_orig    : in std_logic_vector(img_width-1 downto 0);
P_E_data_in_y_orig    : in std_logic_vector(img_height-1 downto 0);
P_E_data_in_fb        : in std_logic;
 
P_E_data_out_pixel     : out std_logic_vector(pix_depth-1 downto 0);    
P_E_data_out_x_dest    : out std_logic_vector(img_width-1 downto 0);    
P_E_data_out_y_dest    : out std_logic_vector(img_height-1 downto 0);   
P_E_data_out_step      : out std_logic_vector(n_steps-1 downto 0);      
P_E_data_out_frame     : out std_logic_vector(n_frames-1 downto 0);     
P_E_data_out_x_orig    : out std_logic_vector(img_width-1 downto 0);    
P_E_data_out_y_orig    : out std_logic_vector(img_height-1 downto 0);   
P_E_data_out_fb        : out std_logic   ;
--Wrapper_E_ready_EB_CF  : in std_logic;
P_E_out_direction      : out std_logic_vector(5 downto 0);
P_E_ack                : out std_logic;
P_E_req                : in std_logic
   
  
   );
end E_wrapper_EB_DEC;

architecture Behavioral of E_wrapper_EB_DEC is


signal	   w_E_ready_DEC_EB             :  std_logic; -- sinal indica se o decoder est? pronto para receber a mensagem 
signal     w_E_x_dest                   :  std_logic_vector(img_width-1 downto 0);
signal     w_E_y_dest                   :  std_logic_vector(img_height-1 downto 0);
signal     w_E_x_orig                   :  std_logic_vector(img_width-1 downto 0);
signal     w_E_y_orig                   :  std_logic_vector(img_height-1 downto 0);
signal     w_E_fb                       :  std_logic;
signal     w_E_out_send                 :  std_logic;
signal     w_E_ready_CF_DEC             :  std_logic;
signal     w_E_out_direction            :  std_logic_vector(5 downto 0);
SIGnal     w_E_valid_EB_DEC   :  std_logic;
SIGnal     w_E_out_new_EB_DEC  :  std_logic;

signal    s_E_data_pixel     :  std_logic_vector(pix_depth-1 downto 0);  
signal    s_E_data_x_dest    :  std_logic_vector(img_width-1 downto 0);  
signal    s_E_data_y_dest    :  std_logic_vector(img_height-1 downto 0); 
signal    s_E_data_step      :  std_logic_vector(n_steps-1 downto 0);    
signal    s_E_data_frame     :  std_logic_vector(n_frames-1 downto 0);   
signal    s_E_data_x_orig    :  std_logic_vector(img_width-1 downto 0);  
signal    s_E_data_y_orig    :  std_logic_vector(img_height-1 downto 0); 
signal    s_E_data_fb        :  std_logic   ;                                                                    
signal    s_E_ack                :  std_logic;                               
signal    s_E_req                : std_logic;                              





      
begin

Inst_dec : entity work.E_dec

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
E_stall_out_DEC      =>   P_E_stall_out_DEC ,
E_stall_in_DEC       =>   P_E_stall_in_DEC  ,
E_in_x_dest          =>  w_E_x_dest,
E_in_y_dest          =>  w_E_y_dest,
E_in_x_orig          =>  w_E_x_orig,
E_in_y_orig          =>  w_E_y_orig,
E_in_fb              =>  w_E_fb    ,
E_out_direction      =>  P_E_out_direction,
E_out_send           => P_E_out_send   ,
E_ready_DEC_EB       => w_E_ready_DEC_EB,
E_valid_EB_DEC       => w_E_valid_EB_DEC,
E_out_new_EB_DEC     => w_E_out_new_EB_DEC

);

Inst_elastic_buffer : entity work.E_elastic_buffer

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
E_valid_EB_DEC         =>     w_E_valid_EB_DEC,
E_ready_DEC_EB         =>     w_E_ready_DEC_EB, 
E_out_new_EB_DEC       =>     w_E_out_new_EB_DEC,  
E_ready_EB_CF          =>     P_E_ready_EB_CF,   
E_stall_in_EB          =>     P_E_stall_in_EB,
E_stall_out_EB         =>     P_E_stall_out_EB,                    
E_data_in_pixel        =>     s_E_data_pixel  ,
E_data_in_x_dest       =>     s_E_data_x_dest ,
E_data_in_y_dest       =>     s_E_data_y_dest ,
E_data_in_step         =>     s_E_data_step   ,
E_data_in_frame        =>     s_E_data_frame  ,
E_data_in_x_orig       =>     s_E_data_x_orig ,
E_data_in_y_orig       =>     s_E_data_y_orig ,
E_data_in_fb           =>     s_E_data_fb     ,
E_ack                  =>     s_E_ack         ,
E_req                  =>     s_E_req         ,
E_data_out_pixel       =>     P_E_data_out_pixel ,
E_data_out_x_dest      =>     w_E_x_dest,
E_data_out_y_dest      =>     w_E_y_dest,
E_data_out_step        =>     P_E_data_out_step  ,
E_data_out_frame       =>     P_E_data_out_frame ,
E_data_out_x_orig      =>     w_E_x_orig,
E_data_out_y_orig      =>     w_E_y_orig,
E_data_out_fb          =>     w_E_fb  
--E_ready_CF_EB          =>     Wrapper_E_ready_EB_CF,
);


Inst_in_E_ctrl : entity work.in_E_controller

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
                                 
oc_E_pixel        =>     P_E_data_in_pixel  ,
oc_E_x_dest       =>     P_E_data_in_x_dest ,
oc_E_y_dest       =>     P_E_data_in_y_dest ,
oc_E_step         =>     P_E_data_in_step   ,
oc_E_frame        =>     P_E_data_in_frame  ,
oc_E_x_orig       =>     P_E_data_in_x_orig,
oc_E_y_orig       =>     P_E_data_in_y_orig,
oc_E_fb           =>     P_E_data_in_fb    ,        
oc_E_new_msg      =>     P_E_req,
oc_E_ack          =>     P_E_ack, 
i_E_pixel         =>     s_E_data_pixel  ,
i_E_x_dest        =>     s_E_data_x_dest ,
i_E_y_dest        =>     s_E_data_y_dest ,
i_E_step          =>     s_E_data_step   ,
i_E_frame         =>     s_E_data_frame  ,
i_E_x_orig        =>     s_E_data_x_orig ,
i_E_y_orig        =>     s_E_data_y_orig ,
i_E_fb            =>     s_E_data_fb     ,
i_E_req           =>     s_E_req         ,
i_E_ack           =>     s_E_ack          

);


P_E_data_out_x_dest <= w_E_x_dest;
P_E_data_out_y_dest <= w_E_y_dest;
P_E_data_out_x_orig <= w_E_x_orig;
P_E_data_out_y_orig <= w_E_y_orig;
P_E_data_out_fb     <= w_E_fb    ;
     
 

   
end Behavioral;
